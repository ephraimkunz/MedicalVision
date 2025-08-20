//
//  DataScanner.swift
//  MedicalVision
//
//  Created by Ephraim Kunz on 8/4/25.
//

import SwiftUI
import VisionKit

struct DataScanner: UIViewControllerRepresentable {
    @Binding var results: [RecognizedItem]
    
    private let scanner: DataScannerViewController
    private let shouldDetect: Bool
    
    init(shouldDetect: Bool, results: Binding<[RecognizedItem]>) {
        self.shouldDetect = shouldDetect
        self.scanner = DataScannerViewController(recognizedDataTypes: [.text()], qualityLevel: .accurate, recognizesMultipleItems: true, isHighlightingEnabled: true)
        self._results = results
    }
    
    func makeUIViewController(context: Context) -> some UIViewController {
        Task {
            for await recognizedItems in scanner.recognizedItems {
                results = recognizedItems
            }
        }
        
        try? scanner.startScanning()
        
        return scanner
    }
    
    func updateUIViewController(_ uiViewController: UIViewControllerType, context: Context) {
        if shouldDetect && !scanner.isScanning {
            try? scanner.startScanning()
        }
        
        if !shouldDetect && scanner.isScanning {
            scanner.stopScanning()
        }
    }
}
