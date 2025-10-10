using Local_Rag_Web.Interfaces;

namespace Local_Rag_Web.Services.DocumentProcessors
{/// <summary>
 /// Factory that returns the appropriate document processor based on file type.
 /// This implements the Factory pattern, which makes it easy to add new 
 /// document types without changing existing code.
 /// </summary>
    public class DocumentProcessorFactory : IDocumentProcessorFactory
    {
        private readonly IEnumerable<IDocumentProcessor> _processors;

        /// <summary>
        /// Constructor receives all registered processors via dependency injection.
        /// This is a common pattern where the DI container automatically
        /// injects all implementations of an interface.
        /// </summary>
        public DocumentProcessorFactory(IEnumerable<IDocumentProcessor> processors)
        {
            _processors = processors ?? throw new ArgumentNullException(nameof(processors));
        }

        public IDocumentProcessor GetProcessor(string fileExtension)
        {
            // Find the first processor that can handle this file extension
            var processor = _processors.FirstOrDefault(p => p.CanProcess(fileExtension));

            if (processor == null)
            {
                throw new NotSupportedException(
                    $"No processor available for file type: {fileExtension}");
            }

            return processor;
        }
    }
}
