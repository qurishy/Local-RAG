using Local_Rag_Web.Models;

namespace Local_Rag_Web.Interfaces
{

    /// <summary>
    /// Service for generating responses using a local LLM.
    /// This creates human-readable answers based on retrieved context.
    /// </summary>
    public interface ILLMService
    {
        /// <summary>
        /// Generates a response to the user's query using retrieved context.
        /// The LLM synthesizes information from multiple sources into a coherent answer.
        /// </summary>
        Task<LLMResponse> GenerateResponseAsync(LLMRequest request);

        /// <summary>
        /// Generates clarifying questions if the user's query is ambiguous.
        /// Helps improve search accuracy by understanding user intent better.
        /// </summary>
        Task<List<string>> GenerateClarifyingQuestionsAsync(string query);
    }
}
