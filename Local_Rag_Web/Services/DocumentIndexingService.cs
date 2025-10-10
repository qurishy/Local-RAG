using Local_Rag_Web.Configuration;
using Local_Rag_Web.DATA;
using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Models;
using Microsoft.Extensions.Options;
using System.Text;
using System.Security.Cryptography;
using Microsoft.EntityFrameworkCore;

namespace Local_Rag_Web.Services
{
    /// <summary>
    /// Orchestrates the entire document indexing pipeline.
    /// 
    /// This service is the heart of building your knowledge base. It takes documents
    /// from the shared folder and transforms them through several stages:
    /// 1. Discovery - Find all supported documents in the folder
    /// 2. Extraction - Pull out the text content from various file formats
    /// 3. Chunking - Break large documents into manageable pieces
    /// 4. Embedding - Convert text chunks into semantic vectors
    /// 5. Storage - Save everything to the database for later retrieval
    /// 
    /// Think of it like a librarian who not only catalogs books but also reads them,
    /// understands their content, and creates a sophisticated index system that allows
    /// finding information by meaning rather than just keywords.
    /// </summary>
    public class DocumentIndexingService : IDocumentIndexingService
    {
        private readonly RAGDbContext _context;
        private readonly IDocumentProcessorFactory _processorFactory;
        private readonly IEmbeddingService _embeddingService;
        private readonly TextChunkingService _chunkingService;
        private readonly RAGSettings _settings;
        private readonly ILogger<DocumentIndexingService> _logger;

        public DocumentIndexingService(
            RAGDbContext context,
            IDocumentProcessorFactory processorFactory,
            IEmbeddingService embeddingService,
            TextChunkingService chunkingService,
            IOptions<RAGSettings> settings,
            ILogger<DocumentIndexingService> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _processorFactory = processorFactory ?? throw new ArgumentNullException(nameof(processorFactory));
            _embeddingService = embeddingService ?? throw new ArgumentNullException(nameof(embeddingService));
            _chunkingService = chunkingService ?? throw new ArgumentNullException(nameof(chunkingService));
            _settings = settings.Value;
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public async Task<int> IndexAllDocumentsAsync(
            IProgress<IndexingProgress> progress = null,
            CancellationToken cancellationToken = default)
        {
            _logger.LogInformation("Starting indexing of all documents in: {Path}",
                _settings.SharedFolderPath);

            // Step 1: Discover all supported documents in the shared folder
            var files = DiscoverDocuments(_settings.SharedFolderPath);

            progress?.Report(new IndexingProgress
            {
                TotalDocuments = files.Count,
                ProcessedDocuments = 0,
                Status = "Starting indexing process..."
            });

            int successCount = 0;

            // Step 2: Process each document
            for (int i = 0; i < files.Count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    _logger.LogWarning("Indexing cancelled by user");
                    break;
                }

                var filePath = files[i];
                var fileName = Path.GetFileName(filePath);

                progress?.Report(new IndexingProgress
                {
                    TotalDocuments = files.Count,
                    ProcessedDocuments = i,
                    CurrentFile = fileName,
                    Status = $"Processing {fileName}..."
                });

                try
                {
                    // Check if this document needs indexing
                    // This prevents re-processing unchanged documents
                    if (!await NeedsReindexingAsync(filePath))
                    {
                        _logger.LogInformation("Skipping unchanged document: {File}", fileName);
                        successCount++;
                        continue;
                    }

                    // Index the document
                    bool success = await IndexDocumentAsync(filePath, cancellationToken);
                    if (success)
                    {
                        successCount++;
                        _logger.LogInformation("Successfully indexed: {File}", fileName);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to index document: {File}", fileName);
                    // Continue with next document even if one fails
                }
            }

            progress?.Report(new IndexingProgress
            {
                TotalDocuments = files.Count,
                ProcessedDocuments = files.Count,
                Status = $"Indexing complete. Successfully processed {successCount} documents."
            });

            _logger.LogInformation("Indexing complete. Processed {Success}/{Total} documents",
                successCount, files.Count);

            return successCount;
        }

        public async Task<bool> IndexDocumentAsync(string filePath, CancellationToken cancellationToken = default)
        {
            if (!File.Exists(filePath))
            {
                _logger.LogWarning("File not found: {Path}", filePath);
                return false;
            }

            try
            {
                // Step 1: Extract text from the document
                var fileExtension = Path.GetExtension(filePath);
                var processor = _processorFactory.GetProcessor(fileExtension);
                var text = await processor.ExtractTextAsync(filePath);

                if (string.IsNullOrWhiteSpace(text))
                {
                    _logger.LogWarning("No text extracted from: {Path}", filePath);
                    return false;
                }

                // Step 2: Break the text into chunks
                // Chunking is crucial because it allows:
                // - Processing documents larger than the model's context window
                // - More precise retrieval (returning relevant paragraphs, not whole documents)
                // - Better embedding quality (embeddings of smaller text are more focused)
                var chunks = _chunkingService.ChunkText(
                    text,
                    _settings.ChunkSize,
                    _settings.ChunkOverlap);

                if (chunks.Count == 0)
                {
                    _logger.LogWarning("No chunks created from: {Path}", filePath);
                    return false;
                }

                // Step 3: Generate embeddings for all chunks
                // This is the most computationally expensive step
                _logger.LogInformation("Generating embeddings for {Count} chunks from: {File}",
                    chunks.Count, Path.GetFileName(filePath));

                var embeddings = await _embeddingService.GenerateEmbeddingsAsync(chunks);

                // Step 4: Calculate file hash to detect changes in the future
                var fileHash = await CalculateFileHashAsync(filePath);

                // Step 5: Store in database
                // We wrap this in a transaction to ensure data consistency
                // If anything fails, the entire operation is rolled back
                using var transaction = await _context.Database.BeginTransactionAsync(cancellationToken);

                try
                {
                    // Check if document already exists and remove it if so
                    // This handles re-indexing of updated documents
                    var existingDoc = await _context.Documents
                        .FirstOrDefaultAsync(d => d.FilePath == filePath, cancellationToken);

                    if (existingDoc != null)
                    {
                        _context.Documents.Remove(existingDoc);
                        await _context.SaveChangesAsync(cancellationToken);
                    }

                    // Create the document record
                    var document = new Document
                    {
                        FileName = Path.GetFileName(filePath),
                        FilePath = filePath,
                        FileType = fileExtension.TrimStart('.'),
                        FileSizeBytes = new FileInfo(filePath).Length,
                        IndexedDate = DateTime.UtcNow,
                        LastModifiedDate = File.GetLastWriteTimeUtc(filePath),
                        FileHash = fileHash
                    };

                    _context.Documents.Add(document);
                    await _context.SaveChangesAsync(cancellationToken);

                    // Create chunk records with their embeddings
                    for (int i = 0; i < chunks.Count; i++)
                    {
                        var chunk = new DocumentChunk
                        {
                            DocumentId = document.Id,
                            Content = chunks[i],
                            ChunkIndex = i,
                            EmbeddingVector = VectorSearchService.SerializeEmbedding(embeddings[i]),
                            TokenCount = _chunkingService.EstimateTokenCount(chunks[i]),
                            CreatedDate = DateTime.UtcNow
                        };

                        _context.DocumentChunks.Add(chunk);
                    }

                    await _context.SaveChangesAsync(cancellationToken);
                    await transaction.CommitAsync(cancellationToken);

                    _logger.LogInformation("Successfully stored document with {Count} chunks: {File}",
                        chunks.Count, Path.GetFileName(filePath));

                    return true;
                }
                catch (Exception ex)
                {
                    await transaction.RollbackAsync(cancellationToken);
                    _logger.LogError(ex, "Database transaction failed for: {Path}", filePath);
                    throw;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to index document: {Path}", filePath);
                return false;
            }
        }

        public async Task<bool> NeedsReindexingAsync(string filePath)
        {
            // Calculate current hash of the file
            var currentHash = await CalculateFileHashAsync(filePath);

            // Check if document exists in database
            var existingDoc = await _context.Documents
                .FirstOrDefaultAsync(d => d.FilePath == filePath);

            if (existingDoc == null)
            {
                // Document is new, needs indexing
                return true;
            }

            // Compare hashes - if different, the file has changed
            return existingDoc.FileHash != currentHash;
        }

        /// <summary>
        /// Discovers all supported documents in the folder and its subdirectories.
        /// This recursively scans the shared folder to find all documents
        /// that we can process based on their file extensions.
        /// </summary>
        private List<string> DiscoverDocuments(string folderPath)
        {
            if (!Directory.Exists(folderPath))
            {
                _logger.LogWarning("Folder does not exist: {Path}", folderPath);
                return new List<string>();
            }

            var documents = new List<string>();

            try
            {
                // Get all files recursively
                // SearchOption.AllDirectories means we search in all subdirectories too
                var allFiles = Directory.GetFiles(folderPath, "*", SearchOption.AllDirectories);

                // Filter to only supported file types
                foreach (var file in allFiles)
                {
                    var extension = Path.GetExtension(file);
                    if (_settings.SupportedFileTypes.Contains(extension, StringComparer.OrdinalIgnoreCase))
                    {
                        documents.Add(file);
                    }
                }

                _logger.LogInformation("Discovered {Count} supported documents in: {Path}",
                    documents.Count, folderPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error discovering documents in: {Path}", folderPath);
            }

            return documents;
        }

        /// <summary>
        /// Calculates a SHA256 hash of the file content.
        /// This hash is like a fingerprint of the file - if even one byte changes,
        /// the hash will be completely different. This allows us to detect when
        /// documents have been modified and need re-indexing.
        /// </summary>
        private async Task<string> CalculateFileHashAsync(string filePath)
        {
            using var stream = File.OpenRead(filePath);
            using var sha256 = SHA256.Create();
            var hashBytes = await sha256.ComputeHashAsync(stream);

            // Convert byte array to hex string
            var stringBuilder = new StringBuilder();
            foreach (var b in hashBytes)
            {
                stringBuilder.Append(b.ToString("x2"));
            }

            return stringBuilder.ToString();
        }
    }
}
