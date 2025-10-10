namespace Local_Rag_Web.Interfaces
{

    /// <summary>
    /// Interface for processing different document types.
    /// Each document type (PDF, Word, etc.) will have its own implementation.
    /// </summary>
    public interface IDocumentProcessor
    {
        /// <summary>
        /// Extracts text content from a document file.
        /// </summary>
        /// <param name="filePath">Full path to the document</param>
        /// <returns>Extracted text content</returns>
        Task<string> ExtractTextAsync(string filePath);

        /// <summary>
        /// Checks if this processor can handle the given file type.
        /// </summary>
        bool CanProcess(string fileExtension);
    }

    /// <summary>
    /// Factory interface for getting the appropriate document processor.
    /// This implements the Strategy pattern for handling different file types.
    /// </summary>
    public interface IDocumentProcessorFactory
    {
        IDocumentProcessor GetProcessor(string fileExtension);
    }
}
