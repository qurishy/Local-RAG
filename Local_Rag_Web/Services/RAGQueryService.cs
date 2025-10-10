using Local_Rag_Web.Configuration;
using Local_Rag_Web.DATA;
using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Local_Rag_Web.Services
{

    /// <summary>
    /// Main orchestration service for the RAG system.
    /// This is the "conductor" that coordinates all the other services
    /// to answer user queries.
    /// 
    /// The RAG pipeline works like this:
    /// 1. User asks a question
    /// 2. We convert the question to an embedding vector (semantic representation)
    /// 3. We search the vector database for similar document chunks
    /// 4. We feed those chunks to the LLM along with the question
    /// 5. The LLM synthesizes an answer based on the retrieved information
    /// 6. We package everything into a report
    /// 
    /// It's called "Retrieval-Augmented Generation" because we're augmenting
    /// the generation (LLM response) with retrieved information (document search).
    /// </summary>
    public class RAGQueryService
    {
        private readonly IEmbeddingService _embeddingService;
        private readonly IVectorSearchService _vectorSearchService;
        private readonly ILLMService _llmService;
        private readonly IReportGenerationService _reportService;
        private readonly RAGDbContext _context;
        private readonly RAGSettings _settings;
        private readonly ILogger<RAGQueryService> _logger;

        public RAGQueryService(
            IEmbeddingService embeddingService,
            IVectorSearchService vectorSearchService,
            ILLMService llmService,
            IReportGenerationService reportService,
            RAGDbContext context,
            IOptions<RAGSettings> settings,
            ILogger<RAGQueryService> logger)
        {
            _embeddingService = embeddingService ?? throw new ArgumentNullException(nameof(embeddingService));
            _vectorSearchService = vectorSearchService ?? throw new ArgumentNullException(nameof(vectorSearchService));
            _llmService = llmService ?? throw new ArgumentNullException(nameof(llmService));
            _reportService = reportService ?? throw new ArgumentNullException(nameof(reportService));
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _settings = settings.Value;
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Main entry point for querying the RAG system.
        /// This method orchestrates the entire pipeline from query to answer.
        /// </summary>
        /// <param name="query">The user's natural language question</param>
        /// <param name="includeReport">Whether to generate a formatted report</param>
        /// <param name="reportFormat">Format for the report (markdown, html, text)</param>
        /// <returns>Query result with answer and optional report</returns>
        public async Task<QueryResult> QueryAsync(
            string query,
            bool includeReport = true,
            ReportFormat reportFormat = ReportFormat.Markdown)
        {
            if (string.IsNullOrWhiteSpace(query))
                throw new ArgumentException("Query cannot be empty", nameof(query));

            _logger.LogInformation("Processing query: {Query}", query);

            try
            {
                // Step 1: Generate embedding for the query
                // This converts the user's question into the same semantic space
                // as our document chunks, allowing us to find similar content
                _logger.LogDebug("Generating query embedding");
                var queryEmbedding = await _embeddingService.GenerateEmbeddingAsync(query);

                // Step 2: Search for relevant document chunks
                // This finds the chunks whose semantic meaning is closest to the query
                _logger.LogDebug("Searching vector database with topK={TopK}, threshold={Threshold}",
                    _settings.TopKResults, _settings.SimilarityThreshold);

                var searchResults = await _vectorSearchService.SearchAsync(
                    queryEmbedding,
                    _settings.TopKResults,
                    _settings.SimilarityThreshold);

                _logger.LogInformation("Found {Count} relevant document chunks", searchResults.Count);

                if (!searchResults.Any())
                {
                    _logger.LogWarning("No relevant documents found for query");
                    return new QueryResult
                    {
                        Query = query,
                        Answer = "I couldn't find any relevant information in the document database to answer your question. " +
                                "The query might be too specific, or the documents might not contain information on this topic.",
                        SearchDate = DateTime.UtcNow,
                        Sources = new List<SourceReference>(),
                        FoundRelevantDocuments = false
                    };
                }

                // Step 3: Prepare context for the LLM
                // We extract the text content from the retrieved chunks
                var contexts = searchResults.Select(r => r.Chunk.Content).ToList();

                // Step 4: Generate response using the LLM
                // The LLM reads through the retrieved chunks and synthesizes an answer
                _logger.LogDebug("Generating LLM response");
                var llmRequest = new LLMRequest
                {
                    Query = query,
                    RetrievedContexts = contexts,
                    MaxTokens = _settings.MaxResponseTokens
                };

                var llmResponse = await _llmService.GenerateResponseAsync(llmRequest);

                _logger.LogInformation("Generated response with {Tokens} tokens", llmResponse.TokensUsed);

                // Step 5: Build source references
                var sources = searchResults.Select(r => new SourceReference
                {
                    FileName = r.Document.FileName,
                    FilePath = r.Document.FilePath,
                    RelevantExcerpt = r.Chunk.Content,
                    SimilarityScore = r.SimilarityScore,
                    ChunkIndex = r.ChunkIndex
                }).ToList();

                // Step 6: Store search history for analytics
                await SaveSearchHistoryAsync(query, searchResults.Count);

                // Step 7: Create the search report object
                var searchReport = new SearchReport
                {
                    Query = query,
                    SearchDate = DateTime.UtcNow,
                    GeneratedAnswer = llmResponse.GeneratedText,
                    Sources = sources,
                    TotalSourcesFound = searchResults.Count,
                    AverageSimilarityScore = searchResults.Average(r => r.SimilarityScore)
                };

                // Step 8: Generate formatted report if requested
                string formattedReport = null;
                if (includeReport)
                {
                    _logger.LogDebug("Generating report in {Format} format", reportFormat);
                    formattedReport = await GenerateReportAsync(searchReport, reportFormat);
                }

                // Step 9: Return the complete result
                return new QueryResult
                {
                    Query = query,
                    Answer = llmResponse.GeneratedText,
                    Sources = sources,
                    SearchDate = searchReport.SearchDate,
                    TotalSourcesFound = searchReport.TotalSourcesFound,
                    AverageSimilarityScore = searchReport.AverageSimilarityScore,
                    FormattedReport = formattedReport,
                    ReportFormat = reportFormat,
                    TokensUsed = llmResponse.TokensUsed,
                    FoundRelevantDocuments = true
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing query: {Query}", query);
                throw new InvalidOperationException($"Failed to process query: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Generates clarifying questions if the user's query is ambiguous.
        /// This helps improve search quality by understanding user intent better.
        /// </summary>
        public async Task<List<string>> GetClarifyingQuestionsAsync(string query)
        {
            _logger.LogInformation("Generating clarifying questions for: {Query}", query);

            try
            {
                var questions = await _llmService.GenerateClarifyingQuestionsAsync(query);
                _logger.LogInformation("Generated {Count} clarifying questions", questions.Count);
                return questions;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate clarifying questions");
                return new List<string>();
            }
        }

        /// <summary>
        /// Gets statistics about the document database.
        /// Useful for monitoring and system health checks.
        /// </summary>
        public async Task<DatabaseStatistics> GetDatabaseStatisticsAsync()
        {
            var totalDocuments = await _context.Documents.CountAsync();
            var totalChunks = await _context.DocumentChunks.CountAsync();
            var recentSearches = await _context.SearchHistories
                .OrderByDescending(s => s.SearchDate)
                .Take(10)
                .ToListAsync();

            var fileTypeStats = await _context.Documents
                .GroupBy(d => d.FileType)
                .Select(g => new FileTypeStatistic
                {
                    FileType = g.Key,
                    Count = g.Count(),
                    TotalSizeBytes = g.Sum(d => d.FileSizeBytes)
                })
                .ToListAsync();

            return new DatabaseStatistics
            {
                TotalDocuments = totalDocuments,
                TotalChunks = totalChunks,
                AverageChunksPerDocument = totalDocuments > 0 ? (double)totalChunks / totalDocuments : 0,
                RecentSearchCount = recentSearches.Count,
                FileTypeStatistics = fileTypeStats,
                LastIndexedDate = await _context.Documents
                    .OrderByDescending(d => d.IndexedDate)
                    .Select(d => d.IndexedDate)
                    .FirstOrDefaultAsync()
            };
        }

        private async Task<string> GenerateReportAsync(SearchReport report, ReportFormat format)
        {
            return format switch
            {
                ReportFormat.Markdown => await _reportService.GenerateMarkdownReportAsync(report),
                ReportFormat.Html => await _reportService.GenerateHtmlReportAsync(report),
                ReportFormat.PlainText => await _reportService.GeneratePlainTextReportAsync(report),
                _ => throw new ArgumentException($"Unsupported report format: {format}")
            };
        }

        private async Task SaveSearchHistoryAsync(string query, int resultCount)
        {
            try
            {
                var searchHistory = new SearchHistory
                {
                    Query = query,
                    SearchDate = DateTime.UtcNow,
                    ResultCount = resultCount
                };

                _context.SearchHistories.Add(searchHistory);
                await _context.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                // Don't fail the query if we can't save history
                _logger.LogError(ex, "Failed to save search history");
            }
        }
    }

    /// <summary>
    /// Result of a RAG query operation.
    /// Contains the answer, sources, and optional formatted report.
    /// </summary>
    public class QueryResult
    {
        public string Query { get; set; }
        public string Answer { get; set; }
        public List<SourceReference> Sources { get; set; }
        public DateTime SearchDate { get; set; }
        public int TotalSourcesFound { get; set; }
        public double AverageSimilarityScore { get; set; }
        public string FormattedReport { get; set; }
        public ReportFormat ReportFormat { get; set; }
        public int TokensUsed { get; set; }
        public bool FoundRelevantDocuments { get; set; }
    }

    /// <summary>
    /// Format options for generated reports.
    /// </summary>
    public enum ReportFormat
    {
        Markdown,
        Html,
        PlainText
    }

    /// <summary>
    /// Statistics about the document database.
    /// </summary>
    public class DatabaseStatistics
    {
        public int TotalDocuments { get; set; }
        public int TotalChunks { get; set; }
        public double AverageChunksPerDocument { get; set; }
        public int RecentSearchCount { get; set; }
        public DateTime? LastIndexedDate { get; set; }
        public List<FileTypeStatistic> FileTypeStatistics { get; set; }
    }

    /// <summary>
    /// Statistics for a specific file type.
    /// </summary>
    public class FileTypeStatistic
    {
        public string FileType { get; set; }
        public int Count { get; set; }
        public long TotalSizeBytes { get; set; }
    }
}
