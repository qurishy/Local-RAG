using Local_Rag_Web.Configuration;
using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

namespace Local_Rag_Web.Services
{
    /// <summary>
    /// Service for generating text responses using a local language model.
    /// 
    /// This is where the "magic" happens - taking retrieved document chunks
    /// and synthesizing them into a coherent, human-readable answer.
    /// 
    /// Think of this as your AI assistant that reads through relevant documents
    /// and writes a clear summary answering the user's question. It's trained
    /// on massive amounts of text and learned patterns of how to explain things,
    /// answer questions, and synthesize information.
    /// 
    /// For a closed network environment, we use models like:
    /// - Phi-3 (Microsoft's small but powerful model, 3.8B parameters)
    /// - TinyLlama (1.1B parameters, very fast)
    /// - Llama 2 7B (larger, more capable, but slower)
    /// 
    /// These can all run locally without internet connection.
    /// </summary>
    public class LLMService : ILLMService, IDisposable
    {
        private readonly InferenceSession _session;
        private readonly RAGSettings _settings;
        private readonly ILogger<LLMService> _logger;
        private readonly LLMTokenizer _tokenizer;

        private const int MAX_CONTEXT_LENGTH = 2048;
        private const string INPUT_IDS_NAME = "input_ids";
        private const string ATTENTION_MASK_NAME = "attention_mask";
        private const string OUTPUT_NAME = "logits";

        public LLMService(
            IOptions<RAGSettings> settings,
            ILogger<LLMService> logger)
        {
            _settings = settings.Value;
            _logger = logger;
            _tokenizer = new LLMTokenizer();

            try
            {
                var sessionOptions = new SessionOptions
                {
                    IntraOpNumThreads = Environment.ProcessorCount,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };

                _session = new InferenceSession(_settings.LLMModelPath, sessionOptions);
                _logger.LogInformation("LLM model loaded successfully from: {Path}",
                    _settings.LLMModelPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load LLM model from: {Path}",
                    _settings.LLMModelPath);
                throw new InvalidOperationException(
                    $"Failed to initialize LLM model from: {_settings.LLMModelPath}", ex);
            }
        }

        public async Task<LLMResponse> GenerateResponseAsync(LLMRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Query))
                throw new ArgumentException("Query cannot be empty", nameof(request));

            try
            {
                var prompt = BuildPrompt(request.Query, request.RetrievedContexts);

                _logger.LogInformation("Generating response for query: {Query}", request.Query);
                _logger.LogDebug("Prompt length: {Length} characters", prompt.Length);

                // FIX: Properly handle MaxTokens - check if it has a value, otherwise use default
                int maxTokens = request.MaxTokens > 0
                    ? request.MaxTokens
                    : _settings.MaxResponseTokens;

                var generatedText = await GenerateTextAsync(prompt, maxTokens);
                var cleanedResponse = CleanGeneratedText(generatedText, prompt);

                return new LLMResponse
                {
                    GeneratedText = cleanedResponse,
                    TokensUsed = _tokenizer.EstimateTokenCount(cleanedResponse)
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate response for query: {Query}", request.Query);
                throw new InvalidOperationException("Failed to generate LLM response", ex);
            }
        }

        public async Task<List<string>> GenerateClarifyingQuestionsAsync(string query)
        {
            var prompt = $@"Based on the following query, generate 2-3 clarifying questions that would help narrow down the search:

Query: {query}

Questions:
1.";

            try
            {
                var response = await GenerateTextAsync(prompt, 150);
                var questions = ParseQuestions(response);
                return questions;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate clarifying questions");
                return new List<string>();
            }
        }

        private string BuildPrompt(string query, List<string> contexts)
        {
            var promptBuilder = new StringBuilder();

            promptBuilder.AppendLine("You are a helpful AI assistant for a defense company's document search system.");
            promptBuilder.AppendLine("Your job is to answer questions based on the provided document excerpts.");
            promptBuilder.AppendLine("Always cite which document excerpt you're using when you answer.");
            promptBuilder.AppendLine("If the provided documents don't contain the answer, clearly state that.");
            promptBuilder.AppendLine();

            if (contexts != null && contexts.Any())
            {
                promptBuilder.AppendLine("Document Excerpts:");
                promptBuilder.AppendLine("---");

                for (int i = 0; i < contexts.Count; i++)
                {
                    promptBuilder.AppendLine($"[Document {i + 1}]");
                    promptBuilder.AppendLine(contexts[i]);
                    promptBuilder.AppendLine();
                }

                promptBuilder.AppendLine("---");
                promptBuilder.AppendLine();
            }

            promptBuilder.AppendLine($"Question: {query}");
            promptBuilder.AppendLine();
            promptBuilder.Append("Answer: ");

            return promptBuilder.ToString();
        }

        private async Task<string> GenerateTextAsync(string prompt, int maxTokens)
        {
            var inputTokens = _tokenizer.Encode(prompt);

            if (inputTokens.Length > MAX_CONTEXT_LENGTH - maxTokens)
            {
                _logger.LogWarning("Prompt too long, truncating from {Original} to {Max} tokens",
                    inputTokens.Length, MAX_CONTEXT_LENGTH - maxTokens);
                inputTokens = inputTokens.Take(MAX_CONTEXT_LENGTH - maxTokens).ToArray();
            }

            var generatedTokens = new List<long>(inputTokens);

            for (int i = 0; i < maxTokens; i++)
            {
                var inputIds = CreateInputTensor(generatedTokens.ToArray());
                var attentionMask = CreateAttentionMask(generatedTokens.Count);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(INPUT_IDS_NAME, inputIds),
                    NamedOnnxValue.CreateFromTensor(ATTENTION_MASK_NAME, attentionMask)
                };

                using var results = await Task.Run(() => _session.Run(inputs));
                var logits = results.First(r => r.Name == OUTPUT_NAME).AsTensor<float>();
                var nextTokenId = SelectNextToken(logits, generatedTokens.Count - 1);

                if (IsEndToken(nextTokenId))
                    break;

                generatedTokens.Add(nextTokenId);

                if (generatedTokens.Count > MAX_CONTEXT_LENGTH)
                    break;
            }

            var generatedText = _tokenizer.Decode(generatedTokens.Skip(inputTokens.Length).ToArray());
            return generatedText;
        }

        private long SelectNextToken(Tensor<float> logits, int position)
        {
            const float temperature = 0.7f;

            var vocabularySize = (int)logits.Dimensions[logits.Dimensions.Length - 1];
            var tokenLogits = new float[vocabularySize];

            for (int i = 0; i < vocabularySize; i++)
            {
                tokenLogits[i] = logits[0, position, i];
            }

            for (int i = 0; i < tokenLogits.Length; i++)
            {
                tokenLogits[i] /= temperature;
            }

            var probabilities = Softmax(tokenLogits);
            return SampleFromDistribution(probabilities);
        }

        private float[] Softmax(float[] logits)
        {
            var maxLogit = logits.Max();
            var expScores = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
            var sumExp = expScores.Sum();

            return expScores.Select(e => (float)(e / sumExp)).ToArray();
        }

        private long SampleFromDistribution(float[] probabilities)
        {
            var random = new Random();
            var randomValue = random.NextDouble();

            double cumulative = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (randomValue <= cumulative)
                    return i;
            }

            return Array.IndexOf(probabilities, probabilities.Max());
        }

        private bool IsEndToken(long tokenId)
        {
            return tokenId == 2 || tokenId == 0;
        }

        private string CleanGeneratedText(string text, string prompt)
        {
            if (text.StartsWith(prompt))
            {
                text = text.Substring(prompt.Length);
            }

            text = text.Trim();

            var endMarkers = new[] { "\n\nQuestion:", "\n\n---", "[Document" };
            foreach (var marker in endMarkers)
            {
                var index = text.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
                if (index > 0)
                {
                    text = text.Substring(0, index);
                }
            }

            return text.Trim();
        }

        private List<string> ParseQuestions(string response)
        {
            var questions = new List<string>();
            var lines = response.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (trimmed.Length > 3 &&
                    (trimmed[0] == '1' || trimmed[0] == '2' || trimmed[0] == '3') &&
                    (trimmed[1] == '.' || trimmed[1] == ')'))
                {
                    var question = trimmed.Substring(2).Trim();
                    if (!string.IsNullOrWhiteSpace(question))
                    {
                        questions.Add(question);
                    }
                }
            }

            return questions;
        }

        private DenseTensor<long> CreateInputTensor(long[] tokens)
        {
            var tensor = new DenseTensor<long>(new[] { 1, tokens.Length });
            for (int i = 0; i < tokens.Length; i++)
            {
                tensor[0, i] = tokens[i];
            }
            return tensor;
        }

        private DenseTensor<long> CreateAttentionMask(int sequenceLength)
        {
            var tensor = new DenseTensor<long>(new[] { 1, sequenceLength });
            for (int i = 0; i < sequenceLength; i++)
            {
                tensor[0, i] = 1;
            }
            return tensor;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    public class LLMTokenizer
    {
        public long[] Encode(string text)
        {
            var words = text.Split(new[] { ' ', '\n', '\r', '\t' },
                StringSplitOptions.RemoveEmptyEntries);

            var tokens = new List<long>();
            foreach (var word in words)
            {
                var hash = word.GetHashCode();
                tokens.Add(Math.Abs(hash % 50000));
            }

            return tokens.ToArray();
        }

        public string Decode(long[] tokens)
        {
            var builder = new StringBuilder();
            foreach (var token in tokens)
            {
                builder.Append($"tok{token} ");
            }

            return builder.ToString().Trim();
        }

        public int EstimateTokenCount(string text)
        {
            return text.Split(new[] { ' ', '\n', '\r', '\t' },
                StringSplitOptions.RemoveEmptyEntries).Length;
        }
    }
}
