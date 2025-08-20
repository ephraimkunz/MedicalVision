//
//  ContentView.swift
//  MedicalVision
//
//  Created by Ephraim Kunz on 8/4/25.
//

import SwiftUI
import VisionKit

struct ContentView: View {
    @AppStorage("extractionMethod") private var nerType: NERType = .coreML
    @AppStorage("nerMinConfidence") private var minConfidence: Double = 0.8

    @State private var liveScannerResults: [RecognizedItem] = []
    @State private var extractionScannerResults: [RecognizedItem] = []
    
    @State private var mode: Mode = .findText
    
    @State private var coreMLNER = CoreMLNER()
    @State private var foundationModelNER: NER =
    if #available(iOS 26, *) {
        FoundationModelNER()
    } else {
        FoundationModelPlaceholderNER()
    }
    
    @State private var showExtractionSettingsModal: Bool = false
    
    var body: some View {
        VStack(alignment: .leading) {
            DataScanner(shouldDetect: mode == .findText, results: $liveScannerResults)
                .ignoresSafeArea(edges: .top)
            
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases) { option in
                    Text(option.name)
                }
            }
            .pickerStyle(.segmented)
            .padding()
            
            ScrollView {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Found Text")
                            .font(.subheadline)
                            .bold()
                        
                        ForEach(scannerResultsForDisplay) {
                            switch $0 {
                            case .text(let text):
                                Text(text.transcript)
                            default:
                                EmptyView()
                            }
                        }
                    }
                    .frame(maxWidth: mode == .findText ? nil : .infinity, alignment: .leading)
                    .padding(.horizontal)
                    
                    if mode == .extractEntities {
                        Divider()
                        
                        VStack(alignment: .leading, spacing: 10) {
                            Button {
                                showExtractionSettingsModal = true
                            } label: {
                                Text("Extracted Entities")
                                    .font(.subheadline)
                                    .bold()
                            }
                            .sheet(isPresented: $showExtractionSettingsModal) {
                                Form {
                                    Picker("Extraction method", selection: $nerType) {
                                        ForEach(NERType.allCases) { option in
                                            Text(option.name)
                                        }
                                    }
                                    .pickerStyle(.segmented)
                                    
                                    switch nerType {
                                    case .coreML:
                                        VStack {
                                            Text("Minimum classification confidence: \(minConfidence, specifier: "%.2f")")
                                                .bold()
                                            Slider(value: $minConfidence, in: 0...1)
                                        }
                                    case .foundationModel:
                                        EmptyView()
                                    }
                                }
                                .presentationDetents(.init(Set([.height(200), .medium])))
                                .presentationDragIndicator(.visible)
                            }
                            
                            switch nerType {
                            case .coreML:
                                ForEach(coreMLNER.items) {
                                    Text($0.text)
                                }
                            case .foundationModel:
                                ForEach(foundationModelNER.items) {
                                    Text($0.text)
                                }
                                
                                if foundationModelNER.isResponding {
                                    ProgressView()
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                    }
                }
            }
            .padding(.horizontal)
        }
        .onChange(of: mode) {
            extractionScannerResults = liveScannerResults
            foundationModelNER.dataScannerResults = extractionScannerResults
            coreMLNER.dataScannerResults = extractionScannerResults
            
            switch nerType {
            case .coreML:
                coreMLNER.enabled = mode == .extractEntities
            case .foundationModel:
                foundationModelNER.enabled = mode == .extractEntities
            }
        }
        .onChange(of: nerType) {
            switch nerType {
            case .coreML:
                coreMLNER.enabled = mode == .extractEntities
            case .foundationModel:
                foundationModelNER.enabled = mode == .extractEntities
            }
        }
        .onChange(of: minConfidence) {
            coreMLNER.minimumConfidence = minConfidence
        }
    }
    
    private var scannerResultsForDisplay: [RecognizedItem] {
        mode == .findText ? liveScannerResults : extractionScannerResults
    }
}

extension RecognizedItem: Equatable {
    public static func == (lhs: RecognizedItem, rhs: RecognizedItem) -> Bool {
        lhs.id == rhs.id
    }
}

enum Mode: CaseIterable, Identifiable {
    case findText
    case extractEntities
    
    var id: Self {
        self
    }
    
    var name: String {
        switch self {
        case .findText:
            "Find Text"
        case .extractEntities:
            "Extract Entities"
        }
    }
}

#Preview {
    ContentView()
}
