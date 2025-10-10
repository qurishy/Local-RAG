using Local_Rag_Web.Models;
using Microsoft.EntityFrameworkCore;

namespace Local_Rag_Web.DATA
{
    /// <summary>
    /// Database context for the RAG system.
    /// This manages all database operations and entity configurations.
    /// </summary>
    public class RAGDbContext : DbContext
    {
        public RAGDbContext(DbContextOptions<RAGDbContext> options)
            : base(options)
        {
        }

        public DbSet<Document> Documents { get; set; }
        public DbSet<DocumentChunk> DocumentChunks { get; set; }
        public DbSet<SearchHistory> SearchHistories { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Configure Document entity
            modelBuilder.Entity<Document>(entity =>
            {
                entity.HasIndex(d => d.FilePath).IsUnique();
                entity.HasIndex(d => d.FileHash);
                entity.HasIndex(d => d.IndexedDate);

                // Configure the one-to-many relationship
                entity.HasMany(d => d.Chunks)
                      .WithOne(c => c.Document)
                      .HasForeignKey(c => c.DocumentId)
                      .OnDelete(DeleteBehavior.Cascade); // When document is deleted, delete its chunks
            });

            // Configure DocumentChunk entity
            modelBuilder.Entity<DocumentChunk>(entity =>
            {
                // The embedding vector can be large, so we specify it explicitly
                entity.Property(c => c.EmbeddingVector)
                      .HasColumnType("varbinary(max)");

                entity.Property(c => c.Content)
                      .HasColumnType("nvarchar(max)");

                entity.HasIndex(c => c.DocumentId);
                entity.HasIndex(c => c.ChunkIndex);
            });

            // Configure SearchHistory entity
            modelBuilder.Entity<SearchHistory>(entity =>
            {
                entity.HasIndex(s => s.SearchDate);
                entity.Property(s => s.Query)
                      .HasColumnType("nvarchar(max)");
            });
        }
    }
}
