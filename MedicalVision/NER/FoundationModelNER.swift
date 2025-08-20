//
//  FoundationModelNER.swift
//  MedicalVision
//
//  Created by Ephraim Kunz on 8/14/25.
//

import Observation
import VisionKit
import FoundationModels

@Observable
@available(iOS 26.0, *) 
class FoundationModelNER: NER {
    private static func newFoundationSession() -> LanguageModelSession {
        LanguageModelSession(model: .default, instructions: "You will be given unstructured text auto-extracted via computer vision from an image a user has taken of a pill bottle label or other medical document. Extract the medical entites from that text. This tagged output will be used to store information about when and how the user should take the medication. As an example, given the prompt \"Funicillin 200MG tablets Take 1-10 tablets by mouth daily as needed.\", the response would be [{recognizedEntity: Medication, sourceText: Funicillin}, {recognizedEntity: Dosage, sourceText: 200MG tablets}, {recognizedEntity: Dosage, sourceText: 1-10 tablets}, {recognizedEntity: Administration, sourceText: by mouth}, {recognizedEntity: Frequency, sourceText: daily as needed}]. That is just an example, the output should not contain those specific entity / source text pairs unless the prompt contains that source text.")
    }
    
    private var foundationSession = newFoundationSession()
    
    private(set) var items: [NERItem] = []
    
    var isResponding: Bool {
        foundationSession.isResponding
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
    
    private func runNERIfEnabled() {
        if enabled {
            runNER()
        }
    }
    
    private func runNER() {
        let text = dataScannerResults.compactMap {
            if case let .text(text) = $0 {
                return text.transcript
            } else {
                return nil
            }
        }.joined(separator: " ")
        
        guard !text.isEmpty else { return }
        
        foundationSession = Self.newFoundationSession()
                
       Task {
           let stream = foundationSession.streamResponse(to: text, generating: [TaggedItem].self)
           
           do {
               for try await partial in stream {
                   self.items = partial.content.map {
                       NERItem(id: UUID(), text: try! AttributedString(markdown: "**\($0.recognizedEntity ?? ""):** \"\($0.sourceText ?? "")\""))
                   }
               }
           } catch {
               var text = AttributedString(error.localizedDescription)
               text.foregroundColor = .red
               self.items = [.init(id: UUID(), text: text)]
           }
        }
    }
}

@Generable
@available(iOS 26.0, *)
private struct TaggedItem: Equatable {
    @Guide(description: "Recognized medical entity")
    @Guide(.anyOf(["Activity", "Administration", "Age", "Area", "Biological Attribute", "Biological Structure", "Clinical Event", "Color", "Coreference", "Date", "Detailed Description", "Diagnostic Procedure", "Disease Disorder", "Distance", "Dosage", "Duration", "Family History", "Frequency", "Height", "History", "Lab Value", "Mass", "Medication", "Nonbiological Location", "Occupation", "Outcome", "Personal Background", "Quantitative Concept", "Severity", "Sex", "Shape", "Sign Symptom", "Subject", "Texture", "Therapeutic Procedure", "Time", "Volume", "Weight"]))
    let recognizedEntity: String
    
    @Guide(description: "Portion of the original text the recognizedEntity maps to")
    let sourceText: String
}
