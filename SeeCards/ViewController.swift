//
//  ViewController.swift
//  SeeCards
//
//  Created by Erik Daniel Haukås Moe on 04/12/2022.
//

import UIKit
import AVFoundation
import CoreML
import Vision

var speechSpeed:Int = 5
var readingInterval:Int = 4
var detectionType:Int = 0

let confidenceThreshold: Float = 0.7  // Only accept detections ≥ 70%


let cardNames: [String: String] = [
    // Clubs
    "2C": "2 of Clubs", "3C": "3 of Clubs", "4C": "4 of Clubs", "5C": "5 of Clubs",
    "6C": "6 of Clubs", "7C": "7 of Clubs", "8C": "8 of Clubs", "9C": "9 of Clubs",
    "10C": "10 of Clubs", "JC": "Jack of Clubs", "QC": "Queen of Clubs", "KC": "King of Clubs", "AC": "Ace of Clubs",

    // Diamonds
    "2D": "2 of Diamonds", "3D": "3 of Diamonds", "4D": "4 of Diamonds", "5D": "5 of Diamonds",
    "6D": "6 of Diamonds", "7D": "7 of Diamonds", "8D": "8 of Diamonds", "9D": "9 of Diamonds",
    "10D": "10 of Diamonds", "JD": "Jack of Diamonds", "QD": "Queen of Diamonds", "KD": "King of Diamonds", "AD": "Ace of Diamonds",

    // Hearts
    "2H": "2 of Hearts", "3H": "3 of Hearts", "4H": "4 of Hearts", "5H": "5 of Hearts",
    "6H": "6 of Hearts", "7H": "7 of Hearts", "8H": "8 of Hearts", "9H": "9 of Hearts",
    "10H": "10 of Hearts", "JH": "Jack of Hearts", "QH": "Queen of Hearts", "KH": "King of Hearts", "AH": "Ace of Hearts",

    // Spades
    "2S": "2 of Spades", "3S": "3 of Spades", "4S": "4 of Spades", "5S": "5 of Spades",
    "6S": "6 of Spades", "7S": "7 of Spades", "8S": "8 of Spades", "9S": "9 of Spades",
    "10S": "10 of Spades", "JS": "Jack of Spades", "QS": "Queen of Spades", "KS": "King of Spades", "AS": "Ace of Spades"
]

let cardNamesNorwegian: [String: String] = [
    // Kløver (Clubs)
    "2C": "kløver to", "3C": "kløver tre", "4C": "kløver fire", "5C": "kløver fem",
    "6C": "kløver seks", "7C": "kløver sju", "8C": "kløver åtte", "9C": "kløver ni",
    "10C": "kløver ti", "JC": "kløver knekt", "QC": "kløver dame", "KC": "kløver konge", "AC": "kløver ess",

    // Ruter (Diamonds)
    "2D": "ruter to", "3D": "ruter tre", "4D": "ruter fire", "5D": "ruter fem",
    "6D": "ruter seks", "7D": "ruter sju", "8D": "ruter åtte", "9D": "ruter ni",
    "10D": "ruter ti", "JD": "ruter knekt", "QD": "ruter dame", "KD": "ruter konge", "AD": "ruter ess",

    // Hjerter (Hearts)
    "2H": "hjerter to", "3H": "hjerter tre", "4H": "hjerter fire", "5H": "hjerter fem",
    "6H": "hjerter seks", "7H": "hjerter sju", "8H": "hjerter åtte", "9H": "hjerter ni",
    "10H": "hjerter ti", "JH": "hjerter knekt", "QH": "hjerter dame", "KH": "hjerter konge", "AH": "hjerter ess",

    // Spar (Spades)
    "2S": "spar to", "3S": "spar tre", "4S": "spar fire", "5S": "spar fem",
    "6S": "spar seks", "7S": "spar sju", "8S": "spar åtte", "9S": "spar ni",
    "10S": "spar ti", "JS": "spar knekt", "QS": "spar dame", "KS": "spar konge", "AS": "spar ess"
]

var activeCardNames = cardNames
var currentLanguage: Int = 0   // 0 = English, 1 = Norwegian

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }

    @IBAction func toggleView(_ sender: Any) {
        
            print("inside camera")
            let secondView = storyboard?.instantiateViewController(identifier: "DeckVC") as! DeckViewController
            secondView.modalPresentationStyle = .fullScreen
            present(secondView, animated: false, completion: nil)
    }
    
    @IBAction func enterSettings(_ sender: Any) {
        
        print("inside how to use")
        let thirdView = storyboard?.instantiateViewController(identifier: "HtuVC") as! HowToUseController
        thirdView.modalPresentationStyle = .fullScreen
        present(thirdView, animated: false, completion: nil)
    }
    
}

class DeckViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate,AVCapturePhotoCaptureDelegate {
    
    var timer: Timer?

    let synthesizer = AVSpeechSynthesizer()
    var speechUtterance: AVSpeechUtterance = AVSpeechUtterance(string:" ")
    
    var detectedLabels = [String]()
    var captureSession = AVCaptureSession()
    var previewView = UIImageView()
    var previewLayer:AVCaptureVideoPreviewLayer!
    var videoOutput:AVCaptureVideoDataOutput!
    var frameCounter = 0
    var frameInterval = 3
    var videoSize = CGSize.zero
    let colors:[UIColor] = {
        var colorSet:[UIColor] = []
        for _ in 0...80 {
            let color = UIColor(red: CGFloat.random(in: 0...1), green: CGFloat.random(in: 0...1), blue: CGFloat.random(in: 0...1), alpha: 1)
            colorSet.append(color)
        }
        return colorSet
    }()
    let ciContext = CIContext()
    var classes:[String] = []
    
    var latestPixelBuffer: CVPixelBuffer?

    var lastDetections = [Detection]()
    
    lazy var yoloRequest:VNCoreMLRequest! = {
        do {
            var model = try stortdataset().model
            switch detectionType {
            case 0: 
                model = try stortdataset().model
            default:
                model = try stortdataset().model
            }
            guard let classes = model.modelDescription.classLabels as? [String] else {
                fatalError()
            }
            self.classes = classes
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel)
            return request
        } catch let error {
            fatalError("mlmodel error.")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupVideo()
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(screenTapped))
        view.addGestureRecognizer(tapGesture)
        //startTimer()
    }
    
    
    @IBAction func goBack(_ sender: Any) {    print("inside main")
        let firstView = storyboard?.instantiateViewController(identifier: "MainDeck") as! ViewController
        firstView.modalPresentationStyle = .fullScreen
        present(firstView, animated: false, completion: nil)
        
        captureSession.stopRunning()
        
        timer?.invalidate()
    }
    
    @objc func screenTapped() {
        guard let pixelBuffer = latestPixelBuffer else {
            print("No frame available yet.")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            guard let drawImage = self.detection(pixelBuffer: pixelBuffer) else { return }

            DispatchQueue.main.async {
                self.previewView.image = drawImage
                self.speakDetectedCards()
            }
        }
    }
    
    @IBAction func readCardsButton(_ sender: UIButton) {
        guard let pixelBuffer = latestPixelBuffer else {
            print("No frame available for detection.")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            guard let drawImage = self.detection(pixelBuffer: pixelBuffer) else { return }

            DispatchQueue.main.async {
                self.previewView.image = drawImage
                self.speakDetectedCards()
            }
        }
    }
    
    func speakDetectedCards() {
        guard !lastDetections.isEmpty else { return }

        // Sort by x-position (left to right)
        let sortedDetections = lastDetections.sorted { $0.box.minX < $1.box.minX }

        // Map to readable card names, then remove duplicates while keeping order
        var seen = Set<String>()
        let texts: [String] = sortedDetections.compactMap { detection in
            guard let label = detection.label else { return nil }
            let name = activeCardNames[label] ?? label
            // Only include the first occurrence
            if seen.contains(name) { return nil }
            seen.insert(name)
            return name
        }

        guard !texts.isEmpty else { return }

        let textToSpeak = texts.joined(separator: ", ")
        let utterance = AVSpeechUtterance(string: textToSpeak)
        
        // Speech speed
        utterance.rate = Float(speechSpeed) / 10.0
        
        // Language
        utterance.voice = AVSpeechSynthesisVoice(language: currentLanguage == 0 ? "en-US" : "nb-NO")
        
        synthesizer.speak(utterance)
    }
    
    func setupVideo(){
        // Use a live video layer instead of a static UIImageView
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.insertSublayer(previewLayer, at: 0)

        captureSession.beginConfiguration()

        let device = AVCaptureDevice.default(for: AVMediaType.video)
        let deviceInput = try! AVCaptureDeviceInput(device: device!)

        captureSession.addInput(deviceInput)
        videoOutput = AVCaptureVideoDataOutput()

        let queue = DispatchQueue(label: "VideoQueue")
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        captureSession.addOutput(videoOutput)
        if let videoConnection = videoOutput.connection(with: .video) {
            if videoConnection.isVideoOrientationSupported {
                videoConnection.videoOrientation = .portrait
            }
        }
        captureSession.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    func detection(pixelBuffer: CVPixelBuffer) -> UIImage? {
        do {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
            try handler.perform([yoloRequest])
            guard let results = yoloRequest.results as? [VNRecognizedObjectObservation] else {
                return nil
            }
            var detections:[Detection] = []
            detectedLabels.removeAll()
            for result in results {
                guard result.confidence >= confidenceThreshold else { continue }

                let flippedBox = CGRect(x: result.boundingBox.minX, y: 1 - result.boundingBox.maxY, width: result.boundingBox.width, height: result.boundingBox.height)
                let box = VNImageRectForNormalizedRect(flippedBox, Int(videoSize.width), Int(videoSize.height))

                guard let label = result.labels.first?.identifier as? String,
                        let colorIndex = classes.firstIndex(of: label) else {
                        return nil
                }
                let detection = Detection(box: box, confidence: result.confidence, label: label, color: colors[colorIndex])
                detections.append(detection)
                detectedLabels.append(activeCardNames[label] ?? label)
            }
            lastDetections = detections
            detectedLabels = Array(Set(detectedLabels))
            let drawImage = drawRectsOnImage(detections, pixelBuffer)
            return drawImage
        } catch let error {
            print(error)
            return nil
        }
    }
    
    func drawRectsOnImage(_ detections: [Detection], _ pixelBuffer: CVPixelBuffer) -> UIImage? {
        print(detectedLabels)
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent)!
        let size = ciImage.extent.size
        guard let cgContext = CGContext(data: nil,
                                        width: Int(size.width),
                                        height: Int(size.height),
                                        bitsPerComponent: 8,
                                        bytesPerRow: 4 * Int(size.width),
                                        space: CGColorSpaceCreateDeviceRGB(),
                                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        cgContext.draw(cgImage, in: CGRect(origin: .zero, size: size))
        for detection in detections {
            let invertedBox = CGRect(x: detection.box.minX, y: size.height - detection.box.maxY, width: detection.box.width, height: detection.box.height)
            if let labelText = detection.label {
                cgContext.textMatrix = .identity
                
                let text = "\(labelText) : \(round(detection.confidence*100))"
                
                let textRect  = CGRect(x: invertedBox.minX + size.width * 0.01, y: invertedBox.minY - size.width * 0.01, width: invertedBox.width, height: invertedBox.height)
                let textStyle = NSMutableParagraphStyle.default.mutableCopy() as! NSMutableParagraphStyle
                
                let textFontAttributes = [
                    NSAttributedString.Key.font: UIFont.systemFont(ofSize: textRect.width * 0.1, weight: .bold),
                    NSAttributedString.Key.foregroundColor: detection.color,
                    NSAttributedString.Key.paragraphStyle: textStyle
                ]
                
                cgContext.saveGState()
                defer { cgContext.restoreGState() }
                let astr = NSAttributedString(string: text, attributes: textFontAttributes)
                let setter = CTFramesetterCreateWithAttributedString(astr)
                let path = CGPath(rect: textRect, transform: nil)
                
                let frame = CTFramesetterCreateFrame(setter, CFRange(), path, nil)
                cgContext.textMatrix = CGAffineTransform.identity
                CTFrameDraw(frame, cgContext)
                
                cgContext.setStrokeColor(detection.color.cgColor)
                cgContext.setLineWidth(9)
                cgContext.stroke(invertedBox)
            }
        }
        
        guard let newImage = cgContext.makeImage() else { return nil }
        return UIImage(ciImage: CIImage(cgImage: newImage))
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if videoSize == CGSize.zero {
            guard let width = sampleBuffer.formatDescription?.dimensions.width,
                  let height = sampleBuffer.formatDescription?.dimensions.height else {
                fatalError()
            }
            videoSize = CGSize(width: CGFloat(width), height: CGFloat(height))
        }

        // Save the latest frame but don’t process it yet
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            latestPixelBuffer = pixelBuffer
        }
    }
    

    @IBAction func goToSettings(_ sender: UIButton) {
        print("inside settings")
        let fourthView = storyboard?.instantiateViewController(identifier: "settingsVC") as! SettingsViewController
        //fourthView.modalPresentationStyle = .fullScreen
        present(fourthView, animated: false, completion: nil)
    }
}

struct Detection {
    let box:CGRect
    let confidence:Float
    let label:String?
    let color:UIColor
}

class SettingsViewController:UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        segmentControl.selectedSegmentIndex = currentLanguage
        
        snakkeHastighet.text = String(describing: speechSpeed)
        
        sliderValue.value = Float(speechSpeed)
        
        //timeInterval.text = String(describing: readingInterval)
        
        //stepperValue.value = Double(readingInterval)
        
        //segmentControl.selectedSegmentIndex = detectionType
    }
    
    
    @IBOutlet weak var segmentControl: UISegmentedControl!
    
    @IBOutlet weak var snakkeHastighet: UILabel!
    
    @IBOutlet weak var sliderValue: UISlider!
    
    //@IBOutlet weak var stepperValue: UIStepper!
    
    //@IBOutlet weak var timeInterval: UILabel!
    
    //@IBAction func incrementing(_ sender: UIStepper) {
    //    timeInterval.text = String(describing: Int(sender.value))
        
    //    readingInterval = Int(sender.value)
    //}
    
    @IBAction func backToMain(_ sender: Any) {
        print("inside main")
        let firstView = storyboard?.instantiateViewController(identifier: "DeckVC") as! DeckViewController
        firstView.modalPresentationStyle = .fullScreen
        present(firstView, animated: false, completion: nil)
    }
    
    @IBAction func sliderChanged(_ sender: Any) {
        snakkeHastighet.text = String(describing: Int(sliderValue.value))
        
        speechSpeed = Int(sliderValue.value)
    }
    
    //@IBAction func toggleChanged(_ sender: UISegmentedControl) {
    //    detectionType = segmentControl.selectedSegmentIndex
    //}
    
    @IBAction func changeLanguage(_ sender: UISegmentedControl) {
        switch sender.selectedSegmentIndex {
        case 0:
            activeCardNames = cardNames
            currentLanguage = 0
        case 1:
            activeCardNames = cardNamesNorwegian
            currentLanguage = 1
        default:
            break
        }
    }
    
    
}

class HowToUseController:UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }

    @IBAction func GoBackToMain(_ sender: UIButton) {
        print("inside main")
            let firstView = storyboard?.instantiateViewController(identifier: "MainDeck") as! ViewController
            firstView.modalPresentationStyle = .fullScreen
            present(firstView, animated: false, completion: nil)
    }
}

