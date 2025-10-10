using Local_Rag_Web.Interfaces;

namespace Local_Rag_Web.Services.DocumentProcessors
{
    /// <summary>
    /// Processes plain text files.
    /// This is the simplest processor since text files don't need parsing.
    /// </summary>
    public class TextDocumentProcessor : IDocumentProcessor
    {
        public bool CanProcess(string fileExtension)
        {
            return fileExtension.Equals(".txt", StringComparison.OrdinalIgnoreCase);
        }

        public async Task<string> ExtractTextAsync(string filePath)
        {
            try
            {
                // Read the entire file as text
                // We use UTF8 encoding which handles most text files correctly
                return await File.ReadAllTextAsync(filePath, System.Text.Encoding.UTF8);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to read text file: {filePath}", ex);
            }
        }
    }
}
