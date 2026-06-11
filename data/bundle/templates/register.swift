// AsiaiRegister — SMAppService registration helper for the Asiai bundle.
//
// Usage: AsiaiRegister register|unregister|status [service|all]
//
// The service -> embedded-plist map is rendered at build time by
// ais_core.bundle (the __SERVICES_MAP__ placeholder below). Status notes:
//   - "requiresApproval" means registration happened but macOS waits for the
//     user to flip the toggle in Settings > Login Items & Extensions.
//   - register() sometimes logs "Operation not permitted" while actually
//     succeeding — trust `status`, not the register error message.
import Foundation
import ServiceManagement

let services: [String: String] = __SERVICES_MAP__

func describe(_ s: SMAppService.Status) -> String {
    switch s {
    case .notRegistered: return "notRegistered"
    case .enabled: return "enabled"
    case .requiresApproval: return "requiresApproval"
    case .notFound: return "notFound"
    @unknown default: return "unknown(\(s.rawValue))"
    }
}

func usage() {
    let names = services.keys.sorted().joined(separator: ", ")
    print("Usage: AsiaiRegister register|unregister|status [service|all]")
    print("Services: \(names)")
}

let args = CommandLine.arguments
guard args.count > 1 else {
    usage()
    exit(2)
}
let action = args[1]
let targetArg = args.count > 2 ? args[2] : "all"

var targets: [(String, String)] = []
if targetArg == "all" {
    targets = services.sorted(by: { $0.key < $1.key }).map { ($0.key, $0.value) }
} else if let plist = services[targetArg] {
    targets = [(targetArg, plist)]
} else {
    print("unknown service \(targetArg)")
    usage()
    exit(2)
}

var rc: Int32 = 0
for (name, plist) in targets {
    let daemon = SMAppService.daemon(plistName: plist)
    switch action {
    case "register":
        do {
            try daemon.register()
            print("\(name): registered (status: \(describe(daemon.status)))")
        } catch {
            print("\(name): register error: \(error) (status: \(describe(daemon.status)))")
            if daemon.status != .enabled && daemon.status != .requiresApproval {
                rc = 1
            }
        }
    case "unregister":
        do {
            try daemon.unregister()
            print("\(name): unregistered")
        } catch {
            print("\(name): unregister error: \(error)")
            rc = 1
        }
    case "status":
        print("\(name): \(describe(daemon.status))")
    default:
        usage()
        exit(2)
    }
}
exit(rc)
