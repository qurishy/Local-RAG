namespace Local_Rag_Web.Models
{
    /// <summary>
    /// Progress information during indexing operations.
    /// </summary>
    public class IndexingProgress
    {
        public int TotalDocuments { get; set; }
        public int ProcessedDocuments { get; set; }
        public string CurrentFile { get; set; }
        public string Status { get; set; }
    }
}
