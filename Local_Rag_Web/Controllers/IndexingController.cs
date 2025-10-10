using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Models;
using Local_Rag_Web.Models.Request;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace Local_Rag_Web.Controllers
{
    /// <summary>
    /// API controller for document indexing operations.
    /// This manages the process of adding documents to the knowledge base.
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class IndexingController : ControllerBase
    {
        private readonly IDocumentIndexingService _indexingService;
        private readonly ILogger<IndexingController> _logger;

        public IndexingController(
            IDocumentIndexingService indexingService,
            ILogger<IndexingController> logger)
        {
            _indexingService = indexingService;
            _logger = logger;
        }

        /// <summary>
        /// Starts indexing all documents in the shared folder.
        /// This is a long-running operation that processes all documents.
        /// </summary>
        [HttpPost("index-all")]
        [ProducesResponseType(typeof(IndexingResult), StatusCodes.Status200OK)]
        public async Task<IActionResult> IndexAllDocuments()
        {
            try
            {
                _logger.LogInformation("Starting document indexing");

                var progress = new Progress<IndexingProgress>(p =>
                {
                    _logger.LogInformation("Indexing progress: {Status}", p.Status);
                });

                var count = await _indexingService.IndexAllDocumentsAsync(progress);

                return Ok(new IndexingResult
                {
                    Success = true,
                    DocumentsProcessed = count,
                    Message = $"Successfully indexed {count} documents"
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during document indexing");
                return StatusCode(500, new IndexingResult
                {
                    Success = false,
                    Message = $"Indexing failed: {ex.Message}"
                });
            }
        }

        /// <summary>
        /// Indexes a single document by file path.
        /// </summary>
        [HttpPost("index-document")]
        [ProducesResponseType(typeof(IndexingResult), StatusCodes.Status200OK)]
        public async Task<IActionResult> IndexDocument([FromBody] IndexDocumentRequest request)
        {
            if (string.IsNullOrWhiteSpace(request?.FilePath))
            {
                return BadRequest(new { error = "File path cannot be empty" });
            }

            try
            {
                var success = await _indexingService.IndexDocumentAsync(request.FilePath);

                return Ok(new IndexingResult
                {
                    Success = success,
                    DocumentsProcessed = success ? 1 : 0,
                    Message = success ? "Document indexed successfully" : "Failed to index document"
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error indexing document: {FilePath}", request.FilePath);
                return StatusCode(500, new IndexingResult
                {
                    Success = false,
                    Message = $"Error: {ex.Message}"
                });
            }
        }

        /// <summary>
        /// Checks if a document needs re-indexing.
        /// </summary>
        [HttpPost("check-reindex")]
        [ProducesResponseType(typeof(ReindexCheckResult), StatusCodes.Status200OK)]
        public async Task<IActionResult> CheckNeedsReindexing([FromBody] IndexDocumentRequest request)
        {
            if (string.IsNullOrWhiteSpace(request?.FilePath))
            {
                return BadRequest(new { error = "File path cannot be empty" });
            }

            try
            {
                var needsReindexing = await _indexingService.NeedsReindexingAsync(request.FilePath);

                return Ok(new ReindexCheckResult
                {
                    FilePath = request.FilePath,
                    NeedsReindexing = needsReindexing
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking reindex status");
                return StatusCode(500, new { error = "An error occurred" });
            }
        }
    }
}