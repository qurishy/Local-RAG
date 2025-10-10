using System.ComponentModel.DataAnnotations;

namespace Local_Rag_Web.Models
{ /// <summary>
  /// Represents a document in the system with its metadata.
  /// This stores the original document information and references to its chunks.
  /// </summary>
    public class Document
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(500)]
        public string FileName { get; set; }

        [Required]
        [MaxLength(2000)]
        public string FilePath { get; set; }

        [Required]
        [MaxLength(50)]
        public string FileType { get; set; } // pdf, docx, txt, etc.

        public long FileSizeBytes { get; set; }

        public DateTime IndexedDate { get; set; }

        public DateTime LastModifiedDate { get; set; }

        // This stores a hash to detect if the document has changed
        [MaxLength(64)]
        public string FileHash { get; set; }

        // Navigation property to related chunks
        public virtual ICollection<DocumentChunk> Chunks { get; set; }
    }
}
