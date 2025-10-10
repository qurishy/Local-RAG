namespace Local_Rag_Web.Models
{ /// <summary>
  /// Request object for LLM generation.
  /// </summary>
    public class LLMRequest
    {
        public string Query { get; set; }
        public List<string> RetrievedContexts { get; set; }
        public int MaxTokens { get; set; }
    }
}
