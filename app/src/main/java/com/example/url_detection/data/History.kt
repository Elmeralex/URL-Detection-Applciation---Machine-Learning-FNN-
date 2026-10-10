package com.example.url_detection.data

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Update
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.flow.Flow

enum class CheckSource(val label: String) { Link("Link"), Qr("QR code") }

/** One check, stored only on the phone. The full URL is kept so Copy and Rescan work later. */
@Entity(tableName = "checks")
data class CheckEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val url: String,
    val source: CheckSource,
    val maliciousProbability: Float,
    val checkedAt: Long,
    val safeBrowsing: SafeBrowsingStatus = SafeBrowsingStatus.NotChecked,
    val threatTypes: String = "", // see encodeThreats
) {
    val verdict: Verdict get() = finalVerdict(maliciousProbability, safeBrowsing)
    val confidence: Float get() = confidenceOf(maliciousProbability)
    val threats: Set<ThreatType> get() = decodeThreats(threatTypes)
}

@Dao
interface CheckDao {
    @Query("SELECT * FROM checks ORDER BY checkedAt DESC")
    fun all(): Flow<List<CheckEntity>>

    @Query("SELECT * FROM checks ORDER BY checkedAt DESC LIMIT :limit")
    fun recent(limit: Int): Flow<List<CheckEntity>>

    @Query("SELECT * FROM checks WHERE id = :id")
    fun byId(id: Long): Flow<CheckEntity?>

    @Insert
    suspend fun insert(check: CheckEntity): Long

    @Update
    suspend fun update(check: CheckEntity)

    @Query("DELETE FROM checks")
    suspend fun clear()
}

// Version 2 adds the Safe Browsing answer. Older checks keep their rows and read as NotChecked.
private val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE checks ADD COLUMN safeBrowsing TEXT NOT NULL DEFAULT 'NotChecked'")
        db.execSQL("ALTER TABLE checks ADD COLUMN threatTypes TEXT NOT NULL DEFAULT ''")
    }
}

@Database(entities = [CheckEntity::class], version = 2, exportSchema = false)
abstract class RestDatabase : RoomDatabase() {
    abstract fun checks(): CheckDao

    companion object {
        @Volatile private var instance: RestDatabase? = null

        fun get(context: Context): RestDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(context.applicationContext, RestDatabase::class.java, "rest.db")
                .addMigrations(MIGRATION_1_2)
                .build()
                .also { instance = it }
        }
    }
}
