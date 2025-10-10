using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Local_Rag_Web.Models.Request;

namespace Local_Rag_Web.Controllers
{
    /// <summary>
    /// Health check controller for monitoring system status.
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class HealthController : ControllerBase
    {
        private readonly ILogger<HealthController> _logger;

        public HealthController(ILogger<HealthController> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Basic health check endpoint.
        /// </summary>
        [HttpGet]
        [ProducesResponseType(typeof(Models.Request.HealthStatus), StatusCodes.Status200OK)]
        public IActionResult Get()
        {
            return Ok(new Models.Request.HealthStatus
            {
                Status = "Healthy",
                Timestamp = DateTime.UtcNow,
                Version = "1.0.0"
            });
        }
    }

}
