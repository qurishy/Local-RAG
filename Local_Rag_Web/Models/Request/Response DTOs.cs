using Local_Rag_Web.Services;

namespace Local_Rag_Web.Models.Request
{
    public class Response_DTOs
    {
    }
    public class SearchRequest
    {
        public string Query { get; set; }
        public bool IncludeReport { get; set; } = true;
        public ReportFormat ReportFormat { get; set; } = ReportFormat.Markdown;
    }

    public class ClarifyRequest
    {
        public string Query { get; set; }
    }

    public class ClarifyingQuestionsResponse
    {
        public string Query { get; set; }
        public List<string> Questions { get; set; }
    }

    public class IndexDocumentRequest
    {
        public string FilePath { get; set; }
    }

    public class IndexingResult
    {
        public bool Success { get; set; }
        public int DocumentsProcessed { get; set; }
        public string Message { get; set; }
    }

    public class ReindexCheckResult
    {
        public string FilePath { get; set; }
        public bool NeedsReindexing { get; set; }
    }

    public class HealthStatus
    {
        public string Status { get; set; }
        public DateTime Timestamp { get; set; }
        public string Version { get; set; }
    }
}
