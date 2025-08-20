//
//  NER.swift
//  MedicalVision
//
//  Created by Ephraim Kunz on 8/4/25.
//

import Observation
import VisionKit
import CoreML
import Tokenizers

@Observable
class CoreMLNER: NER {
    private let model: BiomedicalNER
    private var tokenizer: (any Tokenizer)?
    
    init() {
        try! model = BiomedicalNER(configuration: MLModelConfiguration())
        Task {
            tokenizer = try! await AutoTokenizer.from(pretrained:  "d4data/biomedical-ner-all")
        }
    }
    
    var dataScannerResults: [RecognizedItem] = [] {
        didSet {
            items = []
        }
    }
    
    var enabled: Bool = false {
        didSet {
            runNERIfEnabled()
        }
    }
    
    var minimumConfidence: Double = 0.8 {
        didSet {
            runNERIfEnabled()
        }
    }
    
    // Synchronous, so the caller wouldn't be able to check this.
    let isResponding: Bool = false
    
    private(set) var items: [NERItem] = []
        
    private func runNERIfEnabled() {
        if enabled {
            runNER()
        }
    }
    
    private func runNER() {
        let allInput = dataScannerResults.compactMap {
            switch $0 {
            case .text(let text):
                text.transcript
            default:
                nil
            }
        }.joined(separator: " ")
        
        guard !allInput.isEmpty else {
            items = []
            return
        }
        
        let tokens = tokenizer?.encode(text: allInput) ?? []
        
        let contextLength = 512
        
        let maxTokens = min(tokens.count, contextLength)
        
        let padLength = maxTokens >= contextLength ? 0 : contextLength - maxTokens
        let inputTokens = Array(tokens[0..<maxTokens]) + Array(repeating: 0, count: padLength)
        let attentionMask = Array(repeating: 1, count: maxTokens) + Array(repeating: 0, count: padLength)
        
        let modelInput = BiomedicalNERInput(input_ids: try! makeMultiArray(from: inputTokens), attention_mask: try! makeMultiArray(from: attentionMask))
        let prediction = try! model.prediction(input: modelInput)
        
        let scores = prediction.logits
        let stringTokens = tokenizer?.convertIdsToTokens(inputTokens).compactMap{ $0 } ?? []
        let tokenLabels = getTokenLabels(tokens: stringTokens, scores: scores)
        
        items = tokenLabels.map { (token, label) in
            let niceLabel = label.replacingOccurrences(of: "_", with: " ").capitalized
            return NERItem(id: UUID(), text: try! AttributedString(markdown: "**\(niceLabel):** \"\(token)\""))
        }
    }
    
    private func getTokenLabels(tokens: [String], scores multiArray: MLMultiArray) -> [(String, String)] {
        // Validate shape: (1, 512, 84)
        guard multiArray.shape.count == 3,
              multiArray.shape[0].intValue == 1,
              multiArray.shape[1].intValue == 512,
              multiArray.shape[2].intValue == 84 else {
            fatalError("Unexpected MLMultiArray shape: \(multiArray.shape)")
        }
        
        let labelCount = 84
        let tokenCount = 512
        
        guard tokens.count == 512 else {
            fatalError("wrong token count (\(tokens.count))")
        }
        
        var tokenLabels: [(String, String)] = []
        
        for tokenIndex in 0..<tokenCount {
            let token = tokens[tokenIndex]
            if token == "[CLS]" || token == "[SEP]" || token == "[PAD]" {
                continue
            }
            
            var scores = [Double]()
            for labelIndex in 0..<labelCount {
                let index = tokenIndex * labelCount + labelIndex
                let score = multiArray[index].doubleValue
                scores.append(score)
            }
            
            let probs = softmax(scores)
            
            let (maxProbIndex, maxProb) = probs.enumerated().max(by: { $0.1 < $1.1 }) ?? (0, 0)
            
            if maxProb >= minimumConfidence && maxProbIndex != 0 {
                tokenLabels.append((token, idToLabel[String(maxProbIndex)]!))
            }
        }
        
        // Combine adjacent token labels
        var combinedTokens: [(String, String)] = []
        var currentLabelType: String?
        var currentToken: String = ""
        for (token, label) in tokenLabels {
            let labelType: String
            if let dashIndex = label.firstIndex(of: "-") {
                let nextIndex = label.index(after: dashIndex)
                labelType = String(label[nextIndex...])
            } else {
                labelType = label
            }

            if labelType == currentLabelType {
                if token.starts(with: "##") {
                    currentToken += token.suffix(from: token.index(token.startIndex, offsetBy: 2))
                } else {
                    currentToken += " " + token
                }
            } else {
                if let currentLabelType {
                    combinedTokens.append((currentToken, currentLabelType))
                }
                
                if token.starts(with: "##") {
                    currentToken = String(token.suffix(from: token.index(token.startIndex, offsetBy: 2)))
                } else {
                    currentToken = token
                }
                
                currentLabelType = labelType
            }
        }
        
        if let currentLabelType{
            combinedTokens.append((currentToken, currentLabelType))
        }
                
        return combinedTokens
    }
    
    private func makeMultiArray(from array: [Int]) throws -> MLMultiArray {
        guard array.count == 512 else {
            throw NSError(domain: "makeMultiArray", code: -1, userInfo: [NSLocalizedDescriptionKey: "Expected 512 elements, got \(array.count)"])
        }
        
        let shape: [NSNumber] = [1, 512]
        let multiArray = try MLMultiArray(shape: shape, dataType: .int32)
        
        for (i, value) in array.enumerated() {
            multiArray[i] = NSNumber(value: Int32(value))
        }
        
        return multiArray
    }
    
    private func softmax(_ scores: [Double]) -> [Double] {
        let maxScore = scores.max() ?? 0
        let expScores = scores.map { exp($0 - maxScore) } // for numerical stability
        let sumExp = expScores.reduce(0, +)
        return expScores.map { $0 / sumExp }
    }
    
    private let idToLabel: [String: String] = [
        "0": "O",
        "1": "B-Activity",
        "2": "B-Administration",
        "3": "B-Age",
        "4": "B-Area",
        "5": "B-Biological_attribute",
        "6": "B-Biological_structure",
        "7": "B-Clinical_event",
        "8": "B-Color",
        "9": "B-Coreference",
        "10": "B-Date",
        "11": "B-Detailed_description",
        "12": "B-Diagnostic_procedure",
        "13": "B-Disease_disorder",
        "14": "B-Distance",
        "15": "B-Dosage",
        "16": "B-Duration",
        "17": "B-Family_history",
        "18": "B-Frequency",
        "19": "B-Height",
        "20": "B-History",
        "21": "B-Lab_value",
        "22": "B-Mass",
        "23": "B-Medication",
        "24": "B-Non[biological](Detailed_description",
        "25": "B-Nonbiological_location",
        "26": "B-Occupation",
        "27": "B-Other_entity",
        "28": "B-Other_event",
        "29": "B-Outcome",
        "30": "B-Personal_[back](Biological_structure",
        "31": "B-Personal_background",
        "32": "B-Qualitative_concept",
        "33": "B-Quantitative_concept",
        "34": "B-Severity",
        "35": "B-Sex",
        "36": "B-Shape",
        "37": "B-Sign_symptom",
        "38": "B-Subject",
        "39": "B-Texture",
        "40": "B-Therapeutic_procedure",
        "41": "B-Time",
        "42": "B-Volume",
        "43": "B-Weight",
        "44": "I-Activity",
        "45": "I-Administration",
        "46": "I-Age",
        "47": "I-Area",
        "48": "I-Biological_attribute",
        "49": "I-Biological_structure",
        "50": "I-Clinical_event",
        "51": "I-Color",
        "52": "I-Coreference",
        "53": "I-Date",
        "54": "I-Detailed_description",
        "55": "I-Diagnostic_procedure",
        "56": "I-Disease_disorder",
        "57": "I-Distance",
        "58": "I-Dosage",
        "59": "I-Duration",
        "60": "I-Family_history",
        "61": "I-Frequency",
        "62": "I-Height",
        "63": "I-History",
        "64": "I-Lab_value",
        "65": "I-Mass",
        "66": "I-Medication",
        "67": "I-Nonbiological_location",
        "68": "I-Occupation",
        "69": "I-Other_entity",
        "70": "I-Other_event",
        "71": "I-Outcome",
        "72": "I-Personal_background",
        "73": "I-Qualitative_concept",
        "74": "I-Quantitative_concept",
        "75": "I-Severity",
        "76": "I-Shape",
        "77": "I-Sign_symptom",
        "78": "I-Subject",
        "79": "I-Texture",
        "80": "I-Therapeutic_procedure",
        "81": "I-Time",
        "82": "I-Volume",
        "83": "I-Weight"
    ]
}
