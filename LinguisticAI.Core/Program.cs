//================================================================
// Copyright (c) Coalition of Good-Hearted Engineers
// Advanced System for Text Analysis and AI-powered Categorization
//================================================================

using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;

class LinguisticAI
{
    public class TextData
    {
        public string Text { get; set; }
        public string Category { get; set; }
    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public string Category;
    }

    static void Main()
    {
        Console.WriteLine("Matn kiriting:");
        string text = Console.ReadLine();

        string[] words = Regex.Matches(text, @"\b[a-zA-Z]+\b")
                              .Select(m => m.Value)
                              .ToArray();

        string[] numbers = Regex.Matches(text, @"\d+")
                                .Select(m => m.Value)
                                .ToArray();

        string[] specialChars = Regex.Matches(text, @"[^\w\s]")
                                     .Select(m => m.Value)
                                     .ToArray();

        string[] formulas = Regex.Matches(text, @"[A-Za-z0-9\^\=\+\-\*/\(\)\|∫]+")
                                 .Select(m => m.Value)
                                 .Where(f => f.Contains("=") || f.Contains("∫") || f.Contains("|"))
                                 .ToArray();

        Dictionary<string, int> wordFrequency = words.GroupBy(w => w.ToLower())
                                                     .ToDictionary(g => g.Key, g => g.Count());

        var mostFrequentWords = wordFrequency.OrderByDescending(kvp => kvp.Value)
                                             .Take(5);

        Dictionary<char, int> charFrequency = text.Where(c => !char.IsWhiteSpace(c))
                                                  .GroupBy(c => c)
                                                  .ToDictionary(g => g.Key, g => g.Count());

        Console.WriteLine("So'zlar: " + string.Join(", ", words));
        Console.WriteLine("Sonlar: " + string.Join(", ", numbers));
        Console.WriteLine("Maxsus belgilar: " + string.Join(", ", specialChars));
        Console.WriteLine("Formulalar: " + string.Join("; ", formulas));

        Console.WriteLine("\nEng ko'p ishlatilgan so'zlar:");
        foreach (var kvp in mostFrequentWords)
        {
            Console.WriteLine($"{kvp.Key}: {kvp.Value} marta");
        }

        Console.WriteLine("\nBelgi chastotalari:");
        foreach (var kvp in charFrequency.OrderByDescending(kvp => kvp.Value).Take(10))
        {
            Console.WriteLine($"{kvp.Key}: {kvp.Value} marta");
        }

        var mlContext = new MLContext();

        var data = new[]
        {
            new TextData { Text = "E=mc^2 fizik qonun", Category = "Formula" },
            new TextData { Text = "Sun'iy intellekt (AI) va Mashinani o'rganish (ML) rivojlanadi", Category = "Tech" },
            new TextData { Text = "Bayes teoremasi statistika sohasiga tegishli", Category = "Math" },
            new TextData { Text = "Bugun juda ajoyib kun!", Category = "Casual" }
        };

        var trainData = mlContext.Data.LoadFromEnumerable(data);

        var pipeline = mlContext.Transforms.Conversion
            .MapValueToKey("Category")
            .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Text"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Category", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


        var model = pipeline.Fit(trainData);
        var predictor = mlContext.Model.CreatePredictionEngine<TextData, Prediction>(model);

        var testText = new TextData { Text = text };
        var prediction = predictor.Predict(testText);

        Console.WriteLine($"\nMatn: {testText.Text}");
        Console.WriteLine($"Kategoriya: {prediction.Category}");
    }
}
