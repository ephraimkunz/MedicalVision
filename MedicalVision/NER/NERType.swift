//
//  NERType.swift
//  MedicalVision
//
//  Created by Ephraim Kunz on 8/16/25.
//

enum NERType: Int, CaseIterable, Identifiable {
    case coreML
    case foundationModel
    
    var id: Self {
        self
    }
    
    var name: String {
        switch self {
        case .coreML:
            "CoreML"
        case .foundationModel:
            "Foundation LLM"
        }
    }
}
