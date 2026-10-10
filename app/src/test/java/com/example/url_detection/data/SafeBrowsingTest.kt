package com.example.url_detection.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SafeBrowsingTest {

    private fun hex(s: String) = s.split(' ').filter { it.isNotEmpty() }.map { it.toInt(16).toByte() }.toByteArray()

    // Captured from v5 urls:search for http://testsafebrowsing.appspot.com/s/malware.html
    // and https://www.google.com/: one match (MALWARE, POTENTIALLY_HARMFUL_APPLICATION), cache 300 s.
    private val malwareAnswer = hex(
        "0a 31 0a 2b " +
            "74 65 73 74 73 61 66 65 62 72 6f 77 73 69 6e 67 2e 61 70 70 73 70 6f 74 2e 63 6f 6d " +
            "2f 73 2f 6d 61 6c 77 61 72 65 2e 68 74 6d 6c " +
            "12 02 01 04 " +
            "12 03 08 ac 02"
    )

    @Test
    fun readsRealMatch() {
        val parsed = SearchUrlsResponse.parse(malwareAnswer)
        assertTrue(parsed.matched)
        assertEquals(setOf(ThreatType.Malware, ThreatType.HarmfulApp), parsed.threats)
        assertEquals(300L, parsed.cacheSeconds)
    }

    @Test
    fun noThreatsMeansClean() {
        val parsed = SearchUrlsResponse.parse(hex("12 03 08 ac 02"))
        assertFalse(parsed.matched)
        assertTrue(parsed.threats.isEmpty())
        assertEquals(300L, parsed.cacheSeconds)
    }

    @Test
    fun emptyBodyMeansClean() {
        val parsed = SearchUrlsResponse.parse(ByteArray(0))
        assertFalse(parsed.matched)
        assertEquals(0L, parsed.cacheSeconds)
    }

    @Test
    fun readsUnpackedThreatTypeAndSkipsUnknownFields() {
        // ThreatUrl { threatTypes = 2 (unpacked) }, then an unknown varint field 7.
        val parsed = SearchUrlsResponse.parse(hex("0a 02 10 02 38 05"))
        assertTrue(parsed.matched)
        assertEquals(setOf(ThreatType.SocialEngineering), parsed.threats)
    }

    @Test
    fun unknownThreatCodeStillCountsAsMatch() {
        val parsed = SearchUrlsResponse.parse(hex("0a 03 12 01 63"))
        assertTrue(parsed.matched)
        assertTrue(parsed.threats.isEmpty())
    }

    @Test(expected = IllegalArgumentException::class)
    fun truncatedAnswerThrows() {
        SearchUrlsResponse.parse(malwareAnswer.copyOf(20))
    }

    @Test
    fun threatsRoundTripThroughStorage() {
        val threats = setOf(ThreatType.SocialEngineering, ThreatType.Malware)
        assertEquals(threats, decodeThreats(encodeThreats(threats)))
        assertTrue(decodeThreats("").isEmpty())
    }

    @Test
    fun matchIsAlwaysDangerous() {
        assertEquals(Verdict.Dangerous, finalVerdict(0.01f, SafeBrowsingStatus.Threat))
        assertEquals(Verdict.Dangerous, finalVerdict(0.5f, SafeBrowsingStatus.Threat))
    }

    @Test
    fun anythingElseKeepsTheModelVerdict() {
        val others = SafeBrowsingStatus.entries - SafeBrowsingStatus.Threat
        for (status in others) {
            assertEquals(Verdict.Safe, finalVerdict(0.1f, status))
            assertEquals(Verdict.Suspicious, finalVerdict(0.5f, status))
            assertEquals(Verdict.Dangerous, finalVerdict(0.9f, status))
        }
    }
}
