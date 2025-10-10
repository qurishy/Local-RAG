using Local_Rag_Web.Models;

namespace Local_Rag_Web.Interfaces
{
    /// <summary>
    /// Service for indexing documents into the vector database.
    /// This orchestrates the entire pipeline: extraction → chunking → embedding → storage.
    /// </summary>
    public interface IDocumentIndexingService
    {
        /// <summary>
        /// Indexes all documents in the configured shared folder.
        /// This is the main entry point for building/updating the knowledge base.
        /// </summary>
        Task<int> IndexAllDocumentsAsync(
            IProgress<IndexingProgress> progress = null,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Indexes a single document file.
        /// </summary>
        Task<bool> IndexDocumentAsync(string filePath, CancellationToken cancellationToken = default);

        /// <summary>
        /// Checks if a document needs re-indexing (based on file hash).
        /// This allows incremental updates without re-processing unchanged files.
        /// </summary>
        Task<bool> NeedsReindexingAsync(string filePath);
    }
}
