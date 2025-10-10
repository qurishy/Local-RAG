using DocumentFormat.OpenXml.Wordprocessing;

namespace Local_Rag_Web.Models
{

    /// <summary>
    /// Represents a complete search report with all metadata.
    /// </summary>
    public class SearchReport
    {
        public string Query { get; set; }
        public DateTime SearchDate { get; set; }
        public string GeneratedAnswer { get; set; }
        public List<SourceReference> Sources { get; set; }
        public int TotalSourcesFound { get; set; }
        public double AverageSimilarityScore { get; set; }
    }
}
