using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Local_Rag_Web.Models
{ /// <summary>
  /// Represents a chunk of a document with its embedding vector.
  /// Documents are split into chunks because LLMs have token limits and 
  /// smaller chunks provide more precise retrieval.
  /// </summary>
    public class DocumentChunk
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public int DocumentId { get; set; }

        // The actual text content of this chunk
        [Required]
        public string Content { get; set; }

        // Position of this chunk in the original document (for maintaining order)
        public int ChunkIndex { get; set; }

        // The embedding vector stored as a binary blob for efficiency
        // We'll serialize the float array to bytes
        public byte[] EmbeddingVector { get; set; }

        // For easier debugging and searching
        public int TokenCount { get; set; }

        public DateTime CreatedDate { get; set; }

        // Navigation property back to the parent document
        [ForeignKey("DocumentId")]
        public virtual Document Document { get; set; }
    }
}
