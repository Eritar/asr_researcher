// systap: write macOS system audio to stdout as 16 kHz mono float32 little-endian,
// like `parec` does on Linux.
//
//   systap                       everything that is played
//   systap --bundle com.microsoft.teams2 [--bundle ...]
//                                only apps whose bundle id starts with one of these
//   systap --pid 1234           only this process
//   systap --list                JSON list of processes that use audio, then exit
//
// Uses a Core Audio process tap (macOS 14.2+), so no virtual audio driver is needed.
// The first run asks for "System Audio Recording" permission for your terminal app;
// without it the tap delivers silence.
// Exit codes: 0 = device/process set changed (caller restarts), 1 = error,
//             2 = none of the requested apps is running yet.
//
// Build: swiftc -O -swift-version 5 -o systap systap.swift

import AppKit
import AVFoundation
import CoreAudio
import Foundation

func fail(_ msg: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write("systap: \(msg)\n".data(using: .utf8)!)
    exit(code)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail("\(what) failed (OSStatus \(status))") }
}

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector,
                               mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}

let system = AudioObjectID(kAudioObjectSystemObject)

func readString(_ obj: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = address(selector)
    var ref: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &ref) == noErr, let r = ref else { return nil }
    return r.takeRetainedValue() as String
}

func readUInt32(_ obj: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32 {
    var addr = address(selector)
    var value = UInt32(0)
    var size = UInt32(MemoryLayout<UInt32>.size)
    AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &value)
    return value
}

struct AudioProcess { let id: AudioObjectID; let pid: pid_t; let bundle: String; let playing: Bool }

func audioProcesses() -> [AudioProcess] {
    var addr = address(kAudioHardwarePropertyProcessObjectList)
    var size = UInt32(0)
    guard AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size) == noErr else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids) == noErr else { return [] }
    return ids.map { id in
        var pidAddr = address(kAudioProcessPropertyPID)
        var pid = pid_t(0)
        var psize = UInt32(MemoryLayout<pid_t>.size)
        AudioObjectGetPropertyData(id, &pidAddr, 0, nil, &psize, &pid)
        return AudioProcess(id: id, pid: pid, bundle: readString(id, kAudioProcessPropertyBundleID) ?? "",
                            playing: readUInt32(id, kAudioProcessPropertyIsRunningOutput) != 0)
    }
}

// --- arguments ---------------------------------------------------------------
var bundles: [String] = []
var pids: [pid_t] = []
func wanted(_ p: AudioProcess) -> Bool {
    pids.contains(p.pid) || bundles.contains { p.bundle.hasPrefix($0) }
}
var args = CommandLine.arguments.dropFirst().makeIterator()
while let a = args.next() {
    switch a {
    case "--list":
        let rows = audioProcesses().filter { !$0.bundle.isEmpty }.map {
            ["pid": Int($0.pid), "bundle": $0.bundle, "playing": $0.playing,
             "name": NSRunningApplication(processIdentifier: $0.pid)?.localizedName ?? $0.bundle] as [String: Any]
        }
        let json = try! JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted])
        FileHandle.standardOutput.write(json)
        exit(0)
    case "--bundle":
        guard let b = args.next() else { fail("--bundle needs a value") }
        bundles.append(b)
    case "--pid":
        guard let v = args.next(), let n = pid_t(v) else { fail("--pid needs a number") }
        pids.append(n)
    default:
        fail("unknown argument \(a)")
    }
}

// Default output device UID (the tap is clocked by it).
var outputDevice = AudioObjectID(kAudioObjectUnknown)
var addr = address(kAudioHardwarePropertyDefaultSystemOutputDevice)
var size = UInt32(MemoryLayout<AudioObjectID>.size)
check(AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &outputDevice), "reading default output device")
guard let outputUID = readString(outputDevice, kAudioDevicePropertyDeviceUID) else { fail("reading output device UID") }

// Tap mixed down to stereo, not muting playback: either everything, or only chosen apps.
let tapDescription: CATapDescription
var tappedIDs: Set<AudioObjectID> = []
if bundles.isEmpty && pids.isEmpty {
    tapDescription = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
} else {
    let matches = audioProcesses().filter(wanted)
    if matches.isEmpty { fail("waiting for \((bundles + pids.map { "pid \($0)" }).joined(separator: ", ")) to use audio", code: 2) }
    tappedIDs = Set(matches.map { $0.id })
    tapDescription = CATapDescription(stereoMixdownOfProcesses: matches.map { $0.id })
}
tapDescription.uuid = UUID()
tapDescription.name = "call-assist"
tapDescription.isPrivate = true
tapDescription.muteBehavior = .unmuted
var tapID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateProcessTap(tapDescription, &tapID), "creating process tap (needs macOS 14.2+)")

var tapFormat = AudioStreamBasicDescription()
addr = address(kAudioTapPropertyFormat)
size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
check(AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &tapFormat), "reading tap format")

let aggregateDescription: [String: Any] = [
    kAudioAggregateDeviceNameKey: "call-assist-tap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outputUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapDriftCompensationKey: true,
        kAudioSubTapUIDKey: tapDescription.uuid.uuidString,
    ]],
]
var aggregateID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID),
      "creating aggregate device")

func cleanup() {
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
}

// The tap reports 48 kHz even when the clocking device runs at another rate (e.g. 44.1 kHz
// Bluetooth headsets); the buffers actually arrive at the aggregate device's rate.
var deviceRate = Float64(0)
addr = address(kAudioDevicePropertyNominalSampleRate)
size = UInt32(MemoryLayout<Float64>.size)
if AudioObjectGetPropertyData(aggregateID, &addr, 0, nil, &size, &deviceRate) == noErr, deviceRate > 0 {
    tapFormat.mSampleRate = deviceRate
}
FileHandle.standardError.write("systap: capturing at \(Int(tapFormat.mSampleRate)) Hz via \(outputUID)\n".data(using: .utf8)!)

// When the default output changes (headset plugged in, etc.) exit; the caller restarts us
// so the tap follows the new device and its sample rate.
addr = address(kAudioHardwarePropertyDefaultSystemOutputDevice)
AudioObjectAddPropertyListenerBlock(system, &addr, DispatchQueue.main) { _, _ in
    FileHandle.standardError.write("systap: default output changed, restarting\n".data(using: .utf8)!)
    cleanup()
    exit(0)
}
if !tappedIDs.isEmpty {
    addr = address(kAudioHardwarePropertyProcessObjectList)
    AudioObjectAddPropertyListenerBlock(system, &addr, DispatchQueue.main) { _, _ in
        let now = Set(audioProcesses().filter(wanted).map { $0.id })
        if !now.isSubset(of: tappedIDs) {
            FileHandle.standardError.write("systap: new matching app process, restarting\n".data(using: .utf8)!)
            cleanup()
            exit(0)
        }
    }
}
addr = address(kAudioDevicePropertyNominalSampleRate)
AudioObjectAddPropertyListenerBlock(outputDevice, &addr, DispatchQueue.main) { _, _ in
    FileHandle.standardError.write("systap: output sample rate changed, restarting\n".data(using: .utf8)!)
    cleanup()
    exit(0)
}

guard let inFormat = AVAudioFormat(streamDescription: &tapFormat),
      let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                    channels: 1, interleaved: false),
      let converter = AVAudioConverter(from: inFormat, to: outFormat)
else { cleanup(); fail("unsupported tap format \(tapFormat)") }
converter.downmix = true

signal(SIGPIPE, SIG_IGN)
let queue = DispatchQueue(label: "systap.io")
var procID: AudioDeviceIOProcID?
check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, queue) { _, inData, _, _, _ in
    guard let input = AVAudioPCMBuffer(pcmFormat: inFormat, bufferListNoCopy: inData, deallocator: nil),
          input.frameLength > 0 else { return }
    let capacity = AVAudioFrameCount(Double(input.frameLength) * 16000 / inFormat.sampleRate) + 32
    guard let output = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
    var consumed = false
    var error: NSError?
    converter.convert(to: output, error: &error) { _, status in
        if consumed { status.pointee = .noDataNow; return nil }
        consumed = true
        status.pointee = .haveData
        return input
    }
    let n = Int(output.frameLength) * 4
    if n > 0, write(1, output.floatChannelData![0], n) < 0 {  // reader went away
        cleanup()
        exit(0)
    }
}, "creating IO proc")
check(AudioDeviceStart(aggregateID, procID), "starting capture")

for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { cleanup(); exit(0) }
    source.resume()
    _ = Unmanaged.passRetained(source)
}
dispatchMain()
