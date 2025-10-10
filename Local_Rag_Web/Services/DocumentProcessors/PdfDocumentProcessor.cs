using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using Local_Rag_Web.Interfaces;

namespace Local_Rag_Web.Services.DocumentProcessors
{
    /// <summary>
    /// Processes PDF documents to extract text content.
    /// PDFs can be complex with images, tables, and multiple columns,
    /// so we use iText7 which handles these scenarios well.
    /// </summary>
    public class PdfDocumentProcessor : IDocumentProcessor
    {
        public bool CanProcess(string fileExtension)
        {
            return fileExtension.Equals(".pdf", StringComparison.OrdinalIgnoreCase);
        }

        public async Task<string> ExtractTextAsync(string filePath)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using var pdfReader = new PdfReader(filePath);
                    using var pdfDoc = new PdfDocument(pdfReader);

                    var textBuilder = new System.Text.StringBuilder();

                    // Extract text from each page
                    for (int i = 1; i <= pdfDoc.GetNumberOfPages(); i++)
                    {
                        var page = pdfDoc.GetPage(i);
                        var strategy = new SimpleTextExtractionStrategy();
                        var pageText = PdfTextExtractor.GetTextFromPage(page, strategy);

                        textBuilder.AppendLine(pageText);
                        textBuilder.AppendLine(); // Add spacing between pages
                    }

                    return textBuilder.ToString();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"Failed to extract text from PDF: {filePath}", ex);
                }
            });
        }
    }
}
