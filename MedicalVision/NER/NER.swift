//
//  NER.swift
//  MedicalVision
//
//  Created by Ephraim Kunz on 8/16/25.
//

import Foundation
import VisionKit

protocol NER {
    var dataScannerResults: [RecognizedItem] { get set }
    var enabled: Bool { get set }
    var items: [NERItem] { get }
    var isResponding: Bool { get }
}
