using Local_Rag_Web.Configuration;
using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;

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

        // These constants define the model's input/output structure
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
                // Initialize ONNX Runtime session for the LLM
                var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
                {
                    IntraOpNumThreads = Environment.ProcessorCount,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    // LLMs benefit from graph optimizations
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
                // Step 1: Construct the prompt
                // The prompt is crucial - it tells the LLM what role to play and what to do
                var prompt = BuildPrompt(request.Query, request.RetrievedContexts);

                _logger.LogInformation("Generating response for query: {Query}", request.Query);
                _logger.LogDebug("Prompt length: {Length} characters", prompt.Length);

                // Step 2: Generate the response
                var generatedText = await GenerateTextAsync(
                    prompt,
                    request.MaxTokens ?? _settings.MaxResponseTokens);

                // Step 3: Extract the actual answer from the generated text
                // Sometimes the model includes the prompt in its output, so we clean it up
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
            // This helps when the user's query is vague or ambiguous
            // The LLM can ask questions to narrow down what they're really looking for
            var prompt = $@"Based on the following query, generate 2-3 clarifying questions that would help narrow down the search:

Query: {query}

Questions:
1.";

            try
            {
                var response = await GenerateTextAsync(prompt, 150);

                // Parse the generated questions
                var questions = ParseQuestions(response);

                return questions;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate clarifying questions");
                return new List<string>();
            }
        }

        /// <summary>
        /// Builds a carefully crafted prompt for the LLM.
        /// The prompt engineering is critical - it guides the model's behavior.
        /// 
        /// We use a system message to set the context, then provide retrieved
        /// documents as evidence, and finally ask the question.
        /// </summary>
        private string BuildPrompt(string query, List<string> contexts)
        {
            var promptBuilder = new StringBuilder();

            // System message: Define the AI's role and behavior
            promptBuilder.AppendLine("You are a helpful AI assistant for a defense company's document search system.");
            promptBuilder.AppendLine("Your job is to answer questions based on the provided document excerpts.");
            promptBuilder.AppendLine("Always cite which document excerpt you're using when you answer.");
            promptBuilder.AppendLine("If the provided documents don't contain the answer, clearly state that.");
            promptBuilder.AppendLine();

            // Provide the retrieved contexts
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

            // The actual question
            promptBuilder.AppendLine($"Question: {query}");
            promptBuilder.AppendLine();
            promptBuilder.Append("Answer: ");

            return promptBuilder.ToString();
        }

        /// <summary>
        /// Generates text using the language model.
        /// This implements autoregressive generation: the model predicts one token
        /// at a time, and each prediction influences the next.
        /// 
        /// It's like the model is writing an essay, thinking about each word
        /// as it goes based on what it's written so far.
        /// </summary>
        private async Task<string> GenerateTextAsync(string prompt, int maxTokens)
        {
            // Tokenize the prompt
            var inputTokens = _tokenizer.Encode(prompt);

            // Ensure we don't exceed the model's context length
            if (inputTokens.Length > MAX_CONTEXT_LENGTH - maxTokens)
            {
                _logger.LogWarning("Prompt too long, truncating from {Original} to {Max} tokens",
                    inputTokens.Length, MAX_CONTEXT_LENGTH - maxTokens);
                inputTokens = inputTokens.Take(MAX_CONTEXT_LENGTH - maxTokens).ToArray();
            }

            var generatedTokens = new List<long>(inputTokens);

            // Autoregressive generation loop
            // We generate one token at a time until we hit max length or an end token
            for (int i = 0; i < maxTokens; i++)
            {
                // Prepare inputs for the model
                var inputIds = CreateInputTensor(generatedTokens.ToArray());
                var attentionMask = CreateAttentionMask(generatedTokens.Count);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(INPUT_IDS_NAME, inputIds),
                    NamedOnnxValue.CreateFromTensor(ATTENTION_MASK_NAME, attentionMask)
                };

                // Run inference
                using var results = await Task.Run(() => _session.Run(inputs));

                // Get the logits (raw predictions) for the next token
                var logits = results.First(r => r.Name == OUTPUT_NAME).AsTensor<float>();

                // The model outputs logits for every possible token
                // We need to select the next token from these logits
                var nextTokenId = SelectNextToken(logits, generatedTokens.Count - 1);

                // Check for end of sequence
                if (IsEndToken(nextTokenId))
                    break;

                generatedTokens.Add(nextTokenId);

                // Safety check to prevent infinite loops
                if (generatedTokens.Count > MAX_CONTEXT_LENGTH)
                    break;
            }

            // Decode tokens back to text
            var generatedText = _tokenizer.Decode(generatedTokens.Skip(inputTokens.Length).ToArray());

            return generatedText;
        }

        /// <summary>
        /// Selects the next token from the model's output logits.
        /// We use sampling with temperature to make generation more natural.
        /// 
        /// Temperature controls randomness:
        /// - Low temperature (0.1): Very deterministic, picks most likely tokens
        /// - High temperature (1.5): More creative and random
        /// - Temperature 1.0: Uses the model's natural probability distribution
        /// </summary>
        private long SelectNextToken(Tensor<float> logits, int position)
        {
            const float temperature = 0.7f; // Slightly conservative for factual responses

            // Get logits for the last position (the next token to generate)
            var vocabularySize = logits.Dimensions[logits.Dimensions.Length - 1];
            var tokenLogits = new float[vocabularySize];

            for (int i = 0; i < vocabularySize; i++)
            {
                tokenLogits[i] = logits[0, position, i];
            }

            // Apply temperature scaling
            for (int i = 0; i < tokenLogits.Length; i++)
            {
                tokenLogits[i] /= temperature;
            }

            // Convert logits to probabilities using softmax
            var probabilities = Softmax(tokenLogits);

            // Sample from the probability distribution
            // This is better than always picking the highest probability (greedy decoding)
            // because it allows for more diverse and natural responses
            return SampleFromDistribution(probabilities);
        }

        /// <summary>
        /// Applies the softmax function to convert logits to probabilities.
        /// Softmax ensures all values are between 0 and 1 and sum to 1.
        /// </summary>
        private float[] Softmax(float[] logits)
        {
            var maxLogit = logits.Max();
            var expScores = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
            var sumExp = expScores.Sum();

            return expScores.Select(e => (float)(e / sumExp)).ToArray();
        }

        /// <summary>
        /// Samples a token index from a probability distribution.
        /// This implements categorical sampling.
        /// </summary>
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

            // Fallback to most probable token
            return Array.IndexOf(probabilities, probabilities.Max());
        }

        private bool IsEndToken(long tokenId)
        {
            // Common end-of-sequence token IDs
            // These vary by model, so you'd need to check your specific model's tokenizer
            return tokenId == 2 || tokenId == 0; // 2 is common for </s>, 0 for <pad>
        }

        private string CleanGeneratedText(string text, string prompt)
        {
            // Remove the prompt if it appears in the output
            if (text.StartsWith(prompt))
            {
                text = text.Substring(prompt.Length);
            }

            // Clean up extra whitespace
            text = text.Trim();

            // Stop at common end markers
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
                // Look for numbered questions
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

    /// <summary>
    /// Simplified tokenizer for LLM input/output.
    /// In production, use the exact tokenizer that matches your model
    /// (e.g., SentencePiece for Llama, GPT tokenizer for GPT models).
    /// </summary>
    public class LLMTokenizer
    {
        public long[] Encode(string text)
        {
            // This is a simplified implementation
            // In production, use the proper tokenizer for your model

            // Basic word-level tokenization
            var words = text.Split(new[] { ' ', '\n', '\r', '\t' },
                StringSplitOptions.RemoveEmptyEntries);

            var tokens = new List<long>();
            foreach (var word in words)
            {
                // Convert word to token ID using hash
                // Real tokenizers use vocabulary files
                var hash = word.GetHashCode();
                tokens.Add(Math.Abs(hash % 50000)); // Typical vocabulary size
            }

            return tokens.ToArray();
        }

        public string Decode(long[] tokens)
        {
            // This is a simplified implementation
            // Real tokenizers have vocabulary mappings

            var builder = new StringBuilder();
            foreach (var token in tokens)
            {
                // In production, look up token in vocabulary
                builder.Append($"tok{token} ");
            }

            return builder.ToString().Trim();
        }

        public int EstimateTokenCount(string text)
        {
            // Rough estimation
            return text.Split(new[] { ' ', '\n', '\r', '\t' },
                StringSplitOptions.RemoveEmptyEntries).Length;
        }
    }
}
