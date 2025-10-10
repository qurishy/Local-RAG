using System.ComponentModel.DataAnnotations;

namespace Local_Rag_Web.Models
{
    /// <summary>
    /// Stores user search queries for analytics and improvement
    /// </summary>
    public class SearchHistory
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public string Query { get; set; }

        public DateTime SearchDate { get; set; }

        public int ResultCount { get; set; }

        // Optional: store which documents were most relevant
        public string RelevantDocumentIds { get; set; }
    }
}
