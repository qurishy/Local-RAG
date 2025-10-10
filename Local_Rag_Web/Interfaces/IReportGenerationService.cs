using Local_Rag_Web.Models;

namespace Local_Rag_Web.Interfaces
{
    /// <summary>
    /// Service for generating formatted reports from search results.
    /// Creates professional documents with proper citations and metadata.
    /// </summary>
    public interface IReportGenerationService
    {
        /// <summary>
        /// Generates a comprehensive report in markdown format.
        /// </summary>
        Task<string> GenerateMarkdownReportAsync(SearchReport report);

        /// <summary>
        /// Generates a report in HTML format for web display.
        /// </summary>
        Task<string> GenerateHtmlReportAsync(SearchReport report);

        /// <summary>
        /// Generates a report in plain text format.
        /// </summary>
        Task<string> GeneratePlainTextReportAsync(SearchReport report);
    }
}
