using Local_Rag_Web.Models.Request;
using Local_Rag_Web.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace Local_Rag_Web.Controllers
{
    /// <summary>
    /// API controller for document search operations.
    /// This provides RESTful endpoints for querying the RAG system.
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class SearchController : ControllerBase
    {
        private readonly RAGQueryService _queryService;
        private readonly ILogger<SearchController> _logger;

        public SearchController(
            RAGQueryService queryService,
            ILogger<SearchController> logger)
        {
            _queryService = queryService;
            _logger = logger;
        }

        /// <summary>
        /// Performs a semantic search and generates an answer.
        /// </summary>
        /// <param name="request">Search request containing the query</param>
        /// <returns>Search result with answer and sources</returns>
        [HttpPost("query")]
        [ProducesResponseType(typeof(QueryResult), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> Query([FromBody] SearchRequest request)
        {
            if (string.IsNullOrWhiteSpace(request?.Query))
            {
                return BadRequest(new { error = "Query cannot be empty" });
            }

            try
            {
                _logger.LogInformation("Received search query: {Query}", request.Query);

                var result = await _queryService.QueryAsync(
                    request.Query,
                    request.IncludeReport,
                    request.ReportFormat);

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing search query");
                return StatusCode(500, new { error = "An error occurred while processing your query" });
            }
        }

        /// <summary>
        /// Gets clarifying questions for an ambiguous query.
        /// </summary>
        [HttpPost("clarify")]
        [ProducesResponseType(typeof(ClarifyingQuestionsResponse), StatusCodes.Status200OK)]
        public async Task<IActionResult> GetClarifyingQuestions([FromBody] ClarifyRequest request)
        {
            if (string.IsNullOrWhiteSpace(request?.Query))
            {
                return BadRequest(new { error = "Query cannot be empty" });
            }

            try
            {
                var questions = await _queryService.GetClarifyingQuestionsAsync(request.Query);

                return Ok(new ClarifyingQuestionsResponse
                {
                    Query = request.Query,
                    Questions = questions
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating clarifying questions");
                return StatusCode(500, new { error = "An error occurred" });
            }
        }

        /// <summary>
        /// Gets statistics about the document database.
        /// </summary>
        [HttpGet("statistics")]
        [ProducesResponseType(typeof(DatabaseStatistics), StatusCodes.Status200OK)]
        public async Task<IActionResult> GetStatistics()
        {
            try
            {
                var stats = await _queryService.GetDatabaseStatisticsAsync();
                return Ok(stats);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving statistics");
                return StatusCode(500, new { error = "An error occurred" });
            }
        }
    }

  
}
