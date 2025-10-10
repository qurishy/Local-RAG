using Local_Rag_Web.Configuration;
using Local_Rag_Web.DATA;
using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Services;
using Local_Rag_Web.Services.DocumentProcessors;
using Microsoft.EntityFrameworkCore;
using Serilog;
using Local_Rag_Web.Models.Request;

var builder = WebApplication.CreateBuilder(args);

// ============================================================================
// CONFIGURATION
// ============================================================================

// Configure Serilog for structured logging
Log.Logger = new LoggerConfiguration()
    .ReadFrom.Configuration(builder.Configuration)
    .Enrich.FromLogContext()
    .WriteTo.Console()
    .WriteTo.File("logs/rag-system-.log", rollingInterval: RollingInterval.Day)
    .CreateLogger();

builder.Host.UseSerilog();

// Load RAG settings from appsettings.json
builder.Services.Configure<RAGSettings>(
    builder.Configuration.GetSection("RAGSettings"));


// ============================================================================
// DATABASE CONFIGURATION
// ============================================================================

// Configure Entity Framework with SQL Server
builder.Services.AddDbContext<RAGDbContext>(options =>
{
    var connectionString = builder.Configuration.GetConnectionString("RAGDatabase");
    options.UseSqlServer(connectionString, sqlOptions =>
    {
        // Enable retry on failure for resilience
        sqlOptions.EnableRetryOnFailure(
            maxRetryCount: 5,
            maxRetryDelay: TimeSpan.FromSeconds(30),
            errorNumbersToAdd: null);

        // Set command timeout
        sqlOptions.CommandTimeout(120);
    });

    // Enable sensitive data logging in development
    if (builder.Environment.IsDevelopment())
    {
        options.EnableSensitiveDataLogging();
        options.EnableDetailedErrors();
    }
});


// ============================================================================
// DEPENDENCY INJECTION - CORE SERVICES
// ============================================================================

// Document Processors - Scoped lifetime (created once per request)
// We register all implementations of IDocumentProcessor
builder.Services.AddScoped<IDocumentProcessor, PdfDocumentProcessor>();
builder.Services.AddScoped<IDocumentProcessor, WordDocumentProcessor>();
builder.Services.AddScoped<IDocumentProcessor, TextDocumentProcessor>();

// Document Processor Factory - uses all registered processors
builder.Services.AddScoped<IDocumentProcessorFactory, DocumentProcessorFactory>();

// Text Chunking Service - Scoped
builder.Services.AddScoped<TextChunkingService>();

// Embedding Service - Singleton because it loads a model that should be reused
// The ONNX model is expensive to load, so we want only one instance
builder.Services.AddSingleton<IEmbeddingService, EmbeddingService>();

// Vector Search Service - Scoped (needs DbContext which is scoped)
builder.Services.AddScoped<IVectorSearchService, VectorSearchService>();

// LLM Service - Singleton for the same reason as embedding service
// The LLM model should be loaded once and reused across all requests
builder.Services.AddSingleton<ILLMService, LLMService>();

// Report Generation Service - Scoped
builder.Services.AddScoped<IReportGenerationService, ReportGenerationService>();

// Document Indexing Service - Scoped
builder.Services.AddScoped<IDocumentIndexingService, DocumentIndexingService>();

// Main RAG Query Service - Scoped
builder.Services.AddScoped<RAGQueryService>();



// ============================================================================
// API CONFIGURATION
// ============================================================================

// Add controllers
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        // Configure JSON serialization
        options.JsonSerializerOptions.PropertyNamingPolicy = null; // Use PascalCase
        options.JsonSerializerOptions.WriteIndented = true; // Pretty print in development
    });

// Add API documentation with Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "Local RAG Document Search API",
        Version = "v1",
        Description = "API for semantic document search and retrieval in a closed network environment",
        Contact = new Microsoft.OpenApi.Models.OpenApiContact
        {
            Name = "Defense Company IT Department",
            Email = "it@company.com"
        }
    });

    // Include XML comments if you add them to your controllers
    // var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
    // var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);
    // c.IncludeXmlComments(xmlPath);
});

// Add CORS if needed (for web interface)
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowLocalhost", builder =>
    {
        builder.WithOrigins("http://localhost:3000", "http://localhost:5173")
               .AllowAnyMethod()
               .AllowAnyHeader()
               .AllowCredentials();
    });
});

// Add health checks
builder.Services.AddHealthChecks()
    .AddDbContextCheck<RAGDbContext>("database");



// Add services to the container.
builder.Services.AddControllersWithViews();

var app = builder.Build();


// ============================================================================
// MIDDLEWARE PIPELINE
// ============================================================================

// Use Serilog request logging
app.UseSerilogRequestLogging();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "Local RAG API v1");
        c.RoutePrefix = string.Empty; // Serve Swagger UI at root
    });
}

// Enable CORS
app.UseCors("AllowLocalhost");

// Use HTTPS redirection (important for security even on closed networks)
app.UseHttpsRedirection();

// Authorization (if you add it later)
app.UseAuthorization();

// Map controllers
app.MapControllers();

// Map health check endpoint
app.MapHealthChecks("/health");


// ============================================================================
// DATABASE INITIALIZATION
// ============================================================================

// Ensure database is created and migrations are applied
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    try
    {
        var context = services.GetRequiredService<RAGDbContext>();

        Log.Information("Ensuring database exists and applying migrations...");
        context.Database.Migrate();
        Log.Information("Database ready");
    }
    catch (Exception ex)
    {
        Log.Fatal(ex, "An error occurred while migrating the database");
        throw;
    }
}

// ============================================================================
// START APPLICATION
// ============================================================================

Log.Information("Starting Local RAG Document Search System");
Log.Information("Environment: {Environment}", app.Environment.EnvironmentName);

try
{
    app.Run();
}
catch (Exception ex)
{
    Log.Fatal(ex, "Application terminated unexpectedly");
}
finally
{
    Log.CloseAndFlush();
}