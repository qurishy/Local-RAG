using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using Local_Rag_Web.Interfaces;
using System.Text;

namespace Local_Rag_Web.Services.DocumentProcessors
{
    /// <summary>
    /// Processes Word documents (.docx) to extract text.
    /// We use OpenXML SDK which is Microsoft's official library
    /// for working with Office documents.
    /// </summary>
    public class WordDocumentProcessor : IDocumentProcessor
    {
        public bool CanProcess(string fileExtension)
        {
            return fileExtension.Equals(".docx", StringComparison.OrdinalIgnoreCase) ||
                   fileExtension.Equals(".doc", StringComparison.OrdinalIgnoreCase);
        }

        public async Task<string> ExtractTextAsync(string filePath)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using var wordDoc = WordprocessingDocument.Open(filePath, false);
                    var body = wordDoc.MainDocumentPart?.Document?.Body;

                    if (body == null)
                        return string.Empty;

                    var textBuilder = new StringBuilder();

                    // Extract text from all paragraphs
                    // This maintains document structure and readability
                    foreach (var paragraph in body.Descendants<Paragraph>())
                    {
                        var paragraphText = paragraph.InnerText;
                        if (!string.IsNullOrWhiteSpace(paragraphText))
                        {
                            textBuilder.AppendLine(paragraphText);
                        }
                    }

                    // Also extract text from tables
                    foreach (var table in body.Descendants<Table>())
                    {
                        foreach (var row in table.Descendants<TableRow>())
                        {
                            var rowText = string.Join(" | ",
                                row.Descendants<TableCell>()
                                   .Select(cell => cell.InnerText));
                            textBuilder.AppendLine(rowText);
                        }
                    }

                    return textBuilder.ToString();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"Failed to extract text from Word document: {filePath}", ex);
                }
            });
        }
    }
}
