//
//  FoundationModelPlaceholderNER.swift
//  MedicalVision
//
//  Created by Ephraim Kunz on 8/16/25.
//

import Foundation
import VisionKit
import Observation

@Observable
class FoundationModelPlaceholderNER: NER {
    init() {
    }
    
    var dataScannerResults: [RecognizedItem] = []
    
    var enabled: Bool = false
    
    var items: [NERItem] {
        return [.init(id: UUID(), text: AttributedString("Upgrade to iOS 26 to use the Foundation model"))]
    }
    
    let isResponding: Bool = false
}
