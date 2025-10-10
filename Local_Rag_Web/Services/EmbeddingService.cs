using Local_Rag_Web.Configuration;
using Local_Rag_Web.Interfaces;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Recognizers.Text.Matcher;
using System.Text.RegularExpressions;

namespace Local_Rag_Web.Services
{
    /// <summary>
    /// Service that generates embeddings using an ONNX model.
    /// 
    /// Embeddings work like this: imagine you have a dictionary where every word
    /// is defined by thousands of other words. An embedding is like a coordinate
    /// in a multi-dimensional space where similar meanings are close together.
    /// For example, "car" and "automobile" would be close in this space, while
    /// "car" and "banana" would be far apart.
    /// 
    /// This service uses a pre-trained neural network (in ONNX format) to convert
    /// text into these vector representations. The typical model we'll use is
    /// "all-MiniLM-L6-v2" which produces 384-dimensional vectors and is excellent
    /// for semantic similarity tasks.
    /// </summary>
    public class EmbeddingService : IEmbeddingService, IDisposable
    {
        private readonly InferenceSession _session;
        private readonly RAGSettings _settings;
        private readonly SimpleTokenizer _tokenizer;

        // ONNX models have specific input/output names that we need to know
        private const string InputName = "input_ids";
        private const string AttentionMaskName = "attention_mask";
        private const string OutputName = "sentence_embedding";

        public int EmbeddingDimension => _settings.EmbeddingDimension;

        public EmbeddingService(IOptions<RAGSettings> settings)
        {
            _settings = settings.Value;
            _tokenizer = new SimpleTokenizer();

            try
            {
                // Initialize the ONNX Runtime session
                // This loads the model into memory and prepares it for inference
                var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions
                {
                    // Use all available CPU cores for faster processing
                    IntraOpNumThreads = Environment.ProcessorCount,

                    // Sequential execution is usually faster for embedding models
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                };

                _session = new InferenceSession(_settings.EmbeddingModelPath, sessionOptions);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to load embedding model from: {_settings.EmbeddingModelPath}", ex);
            }
        }

        public async Task<float[]> GenerateEmbeddingAsync(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be empty", nameof(text));

            // Tokenization: Convert text into numeric IDs that the model understands
            // This is like translating English into a language the neural network speaks
            var tokens = _tokenizer.Tokenize(text);

            // Create input tensors for the ONNX model
            // Think of tensors as multi-dimensional arrays that neural networks process
            var inputTensor = CreateInputTensor(tokens);
            var attentionMask = CreateAttentionMask(tokens.Length);

            // Prepare the input for the model
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(InputName, inputTensor),
                NamedOnnxValue.CreateFromTensor(AttentionMaskName, attentionMask)
            };

            // Run inference - this is where the magic happens
            // The model processes the tokens and produces an embedding vector
            using var results = await Task.Run(() => _session.Run(inputs));

            // Extract the embedding from the output
            var outputTensor = results.First(r => r.Name == OutputName).AsTensor<float>();

            // The output is a 2D tensor [batch_size, embedding_dim]
            // We only have one input, so we take the first row
            var embedding = new float[EmbeddingDimension];
            for (int i = 0; i < EmbeddingDimension; i++)
            {
                embedding[i] = outputTensor[0, i];
            }

            // Normalize the embedding vector
            // This makes all vectors have length 1, which is important for
            // cosine similarity calculations later
            return NormalizeVector(embedding);
        }

        public async Task<List<float[]>> GenerateEmbeddingsAsync(List<string> texts)
        {
            var embeddings = new List<float[]>();

            // Process each text
            // In a production system, you might want to implement true batch processing
            // for better performance, but this approach is simpler and works well
            // for most use cases
            foreach (var text in texts)
            {
                try
                {
                    var embedding = await GenerateEmbeddingAsync(text);
                    embeddings.Add(embedding);
                }
                catch (Exception ex)
                {
                    // Log the error but continue processing other texts
                    Console.WriteLine($"Failed to generate embedding for text: {ex.Message}");
                    // Add a zero vector as a placeholder
                    embeddings.Add(new float[EmbeddingDimension]);
                }
            }

            return embeddings;
        }

        /// <summary>
        /// Creates the input tensor from tokenized text.
        /// </summary>
        private DenseTensor<long> CreateInputTensor(int[] tokens)
        {
            // Create a 2D tensor with shape [batch_size, sequence_length]
            // batch_size is 1 because we process one text at a time
            var tensor = new DenseTensor<long>(new[] { 1, tokens.Length });

            for (int i = 0; i < tokens.Length; i++)
            {
                tensor[0, i] = tokens[i];
            }

            return tensor;
        }

        /// <summary>
        /// Creates the attention mask tensor.
        /// The attention mask tells the model which tokens are real
        /// and which are padding (if we needed to pad to a fixed length).
        /// </summary>
        private DenseTensor<long> CreateAttentionMask(int sequenceLength)
        {
            var tensor = new DenseTensor<long>(new[] { 1, sequenceLength });

            // All ones means "pay attention to all tokens"
            for (int i = 0; i < sequenceLength; i++)
            {
                tensor[0, i] = 1;
            }

            return tensor;
        }

        /// <summary>
        /// Normalizes a vector to unit length.
        /// After normalization, the vector has length 1, which makes
        /// cosine similarity calculations simpler and more numerically stable.
        /// </summary>
        private float[] NormalizeVector(float[] vector)
        {
            // Calculate the magnitude (length) of the vector
            // This is the square root of the sum of squares
            double magnitude = Math.Sqrt(vector.Sum(v => v * v));

            if (magnitude == 0)
                return vector;

            // Divide each component by the magnitude
            var normalized = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                normalized[i] = (float)(vector[i] / magnitude);
            }

            return normalized;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    /// <summary>
    /// A simple tokenizer that converts text to token IDs.
    /// 
    /// Important note: In a production system, you should use the exact same
    /// tokenizer that was used to train your embedding model. For models like
    /// all-MiniLM-L6-v2, this would be the BERT tokenizer.
    /// 
    /// This implementation is simplified. For better results, consider using
    /// the Microsoft.ML.Tokenizers NuGet package which provides proper
    /// WordPiece tokenization that matches BERT models.
    /// </summary>
    public class SimpleTokenizer
    {
        // Special tokens used by BERT-style models
        private const int CLS_TOKEN_ID = 101;  // [CLS] - start of sequence
        private const int SEP_TOKEN_ID = 102;  // [SEP] - end of sequence
        private const int UNK_TOKEN_ID = 100;  // [UNK] - unknown token
        private const int MAX_SEQUENCE_LENGTH = 512;

        public int[] Tokenize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new[] { CLS_TOKEN_ID, SEP_TOKEN_ID };

            // Clean and normalize the text
            text = text.ToLowerInvariant();
            text = Regex.Replace(text, @"\s+", " ").Trim();

            // Split into words
            var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // Convert words to token IDs
            // In a real implementation, you'd use a vocabulary file
            // that maps words to their specific IDs
            var tokenIds = new List<int> { CLS_TOKEN_ID };

            foreach (var word in words)
            {
                // This is a simplified approach
                // A real tokenizer would use WordPiece or similar subword tokenization
                var tokenId = GetTokenId(word);
                tokenIds.Add(tokenId);

                // Respect maximum sequence length
                if (tokenIds.Count >= MAX_SEQUENCE_LENGTH - 1)
                    break;
            }

            tokenIds.Add(SEP_TOKEN_ID);

            return tokenIds.ToArray();
        }

        /// <summary>
        /// Gets a token ID for a word.
        /// This is simplified - use a proper vocabulary in production!
        /// </summary>
        private int GetTokenId(string word)
        {
            // A real implementation would look up the word in a vocabulary
            // For now, we'll use a hash-based approach that's consistent
            // but not ideal for production use

            // NOTE: For production, load the vocabulary from vocab.txt file
            // that comes with your BERT model

            int hash = word.GetHashCode();
            // Map to a reasonable range (30000 is typical vocab size)
            return Math.Abs(hash % 30000) + 1000;
        }
    }

}
