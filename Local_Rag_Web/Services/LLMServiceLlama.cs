using LLama;
using LLama.Common;
using Local_Rag_Web.Configuration;
using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Models;
using Microsoft.Extensions.Options;

using System.Text;

namespace Local_Rag_Web.Services
{
    /// <summary>
    /// Simplified LLamaSharp service that works with most versions.
    /// Uses StatelessExecutor for simpler, more reliable generation.
    /// </summary>
    public class LLMServiceLlama : ILLMService, IDisposable
    {
        private readonly LLamaWeights _model;
        private readonly LLamaContext _context;
        private readonly RAGSettings _settings;
        private readonly ILogger<LLMServiceLlama> _logger;

        public LLMServiceLlama(
            IOptions<RAGSettings> settings,
            ILogger<LLMServiceLlama> logger)
        {
            _settings = settings.Value;
            _logger = logger;

            try
            {
                _logger.LogInformation("Loading Llama model from: {Path}", _settings.LLMModelPath);

                // Simple model parameters
                var parameters = new ModelParams(_settings.LLMModelPath)
                {
                    ContextSize = 2048,
                    GpuLayerCount = 0
                };

                _model = LLamaWeights.LoadFromFile(parameters);
                _context = _model.CreateContext(parameters);

                _logger.LogInformation("✅ Llama model loaded successfully!");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to load Llama model");
                throw new InvalidOperationException(
                    $"Failed to initialize Llama model from: {_settings.LLMModelPath}", ex);
            }
        }

        public async Task<LLMResponse> GenerateResponseAsync(LLMRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Query))
                throw new ArgumentException("Query cannot be empty", nameof(request));

            try
            {
                _logger.LogInformation("🤖 Generating response for: {Query}", request.Query);

                var prompt = BuildPrompt(request.Query, request.RetrievedContexts);
                var maxTokens = request.MaxTokens > 0 ? request.MaxTokens : _settings.MaxResponseTokens;

                // Use StatelessExecutor - simpler and more compatible
                var executor = new StatelessExecutor(_model, _context.Params);

                // Simple inference parameters
                var inferenceParams = new InferenceParams
                {
                    MaxTokens = maxTokens,
                    AntiPrompts = new List<string> { "[INST]", "Question:", "\n\n" }
                };

                var response = new StringBuilder();
                var tokenCount = 0;

                await foreach (var token in executor.InferAsync(prompt, inferenceParams))
                {
                    response.Append(token);
                    tokenCount++;

                    // Safety limit
                    if (tokenCount >= maxTokens)
                        break;
                }

                var generatedText = response.ToString();
                var cleanedText = CleanResponse(generatedText);

                _logger.LogInformation("✅ Generated {Tokens} tokens", tokenCount);

                return new LLMResponse
                {
                    GeneratedText = cleanedText,
                    TokensUsed = tokenCount
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to generate response");

                // Return a graceful fallback
                return new LLMResponse
                {
                    GeneratedText = GenerateFallbackResponse(request.Query, request.RetrievedContexts),
                    TokensUsed = 0
                };
            }
        }

        public async Task<List<string>> GenerateClarifyingQuestionsAsync(string query)
        {
            try
            {
                var prompt = $"Generate 3 clarifying questions about: {query}\n\n1.";

                var executor = new StatelessExecutor(_model, _context.Params);
                var inferenceParams = new InferenceParams
                {
                    MaxTokens = 100,
                    AntiPrompts = new List<string> { "\n\n" }
                };

                var response = new StringBuilder();
                await foreach (var token in executor.InferAsync(prompt, inferenceParams))
                {
                    response.Append(token);
                }

                var questions = ParseQuestions(response.ToString());

                if (questions.Any())
                    return questions;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate clarifying questions");
            }

            // Fallback questions
            return new List<string>
            {
                $"What specific information about '{query}' do you need?",
                $"Are you looking for technical details or an overview?",
                $"Do you need current information or historical context?"
            };
        }

        private string BuildPrompt(string query, List<string> contexts)
        {
            var sb = new StringBuilder();

            // Llama-2 chat template
            sb.AppendLine("[INST] <<SYS>>");
            sb.AppendLine("You are a helpful assistant. Answer the question based on the provided documents.");
            sb.AppendLine("Be concise and specific. Cite which document you're referencing.");
            sb.AppendLine("<</SYS>>");
            sb.AppendLine();

            // Add context if available
            if (contexts != null && contexts.Any())
            {
                sb.AppendLine("Documents:");

                for (int i = 0; i < Math.Min(3, contexts.Count); i++)
                {
                    sb.AppendLine($"\n[Doc {i + 1}]");

                    // Limit context size
                    var ctx = contexts[i];
                    if (ctx.Length > 400)
                        ctx = ctx.Substring(0, 397) + "...";

                    sb.AppendLine(ctx);
                }

                sb.AppendLine();
            }

            sb.AppendLine($"Question: {query}");
            sb.AppendLine("[/INST]");
            sb.AppendLine("Answer: ");

            return sb.ToString();
        }

        private string CleanResponse(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return "Unable to generate response.";

            // Remove formatting artifacts
            text = text.Replace("[INST]", "")
                      .Replace("[/INST]", "")
                      .Replace("<<SYS>>", "")
                      .Replace("<</SYS>>", "")
                      .Replace("Answer: ", "")
                      .Trim();

            // Stop at unwanted content
            var stopWords = new[] { "\nQuestion:", "\n[Doc", "\nDocuments:" };
            foreach (var stop in stopWords)
            {
                var idx = text.IndexOf(stop, StringComparison.OrdinalIgnoreCase);
                if (idx > 0)
                    text = text.Substring(0, idx);
            }

            return text.Trim();
        }

        private string GenerateFallbackResponse(string query, List<string> contexts)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Based on the search for '{query}', here are the relevant excerpts:");
            sb.AppendLine();

            if (contexts != null && contexts.Any())
            {
                for (int i = 0; i < Math.Min(3, contexts.Count); i++)
                {
                    sb.AppendLine($"From Document {i + 1}:");
                    var excerpt = contexts[i];
                    if (excerpt.Length > 300)
                        excerpt = excerpt.Substring(0, 297) + "...";
                    sb.AppendLine(excerpt);
                    sb.AppendLine();
                }
            }
            else
            {
                sb.AppendLine("No relevant documents found.");
            }

            return sb.ToString();
        }

        private List<string> ParseQuestions(string text)
        {
            var questions = new List<string>();
            var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (trimmed.Length > 10 &&
                    char.IsDigit(trimmed[0]) &&
                    trimmed[1] == '.')
                {
                    var question = trimmed.Substring(2).Trim();
                    if (!string.IsNullOrWhiteSpace(question))
                        questions.Add(question);
                }
            }

            return questions.Take(3).ToList();
        }

        public void Dispose()
        {
            _context?.Dispose();
            _model?.Dispose();
        }
    }
}
