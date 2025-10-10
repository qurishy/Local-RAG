namespace Local_Rag_Web.Models
{
    /// <summary>
    /// Response from LLM generation.
    /// </summary>
    public class LLMResponse
    {
        public string GeneratedText { get; set; }
        public int TokensUsed { get; set; }
    }

}
