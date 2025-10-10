namespace Local_Rag_Web.Models
{
    /// <summary>
    /// Reference to a source document used in the answer.
    /// </summary>
    public class SourceReference
    {
        public string FileName { get; set; }
        public string FilePath { get; set; }
        public string RelevantExcerpt { get; set; }
        public double SimilarityScore { get; set; }
        public int ChunkIndex { get; set; }
    }
}
