# Dataset: AAA Multiplayer Networking & Replication Architecture
**Category:** Multiplayer Systems Engineering  
**Target:** AI Model Training / Senior Technical Design  
**Framework Focus:** Unreal Engine 5 (UE5), C++20, Iris Replication System, Data-Oriented Design (ECS)

---

## 1. Architectural Foundations: The Authoritative Server Model

In AAA development, the **Authoritative Server** model is the industry standard for competitive integrity and state consistency.

### 1.1 Actor Roles and Ownership
Every Actor in a networked environment exists in a specific local context defined by its `ENetRole`.
- **`ROLE_Authority`**: The server-side instance. The only instance allowed to mutate "true" state.
- **`ROLE_AutonomousProxy`**: The local player’s representation on their own client. Can perform **Client-Side Prediction**.
- **`ROLE_SimulatedProxy`**: The representation of other players/entities on a remote client. State is updated via **Interpolation**.

### 1.2 The Net Driver & Packet Flow
The `UNetDriver` handles the serialization of data into bitstreams.
1. **Property Replication**: Pushing state from Server -> Client.
2. **RPCs (Remote Procedure Calls)**: Messaging between Client and Server.
3. **Relevancy & Prioritization**: Determining *what* needs to be sent to *whom* to save bandwidth.

---

## 2. Advanced C++ Replication Implementation

### 2.1 Property Replication with Logic
Standard replication uses the `GetLifetimeReplicatedProps` function. For performance, minimize the frequency of updates.

```cpp
// Header (.h)
UPROPERTY(ReplicatedUsing = OnRep_Health)
float Health;

UFUNCTION()
void OnRep_Health(float OldHealth);

// Implementation (.cpp)
void ABaseCharacter::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const {
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);

    // DOREPLIFETIME_CONDITION saves bandwidth by only sending to relevant clients
    DOREPLIFETIME_CONDITION(ABaseCharacter, Health, COND_None); 
    DOREPLIFETIME_CONDITION(ABaseCharacter, Mana, COND_OwnerOnly);
}

void ABaseCharacter::OnRep_Health(float OldHealth) {
    if (Health < OldHealth) {
        PlayHitNotify(); // Visual feedback triggered by state change
    }
}
```

### 2.2 Reliable vs. Unreliable RPCs
- **Reliable**: Guaranteed arrival (e.g., Purchasing an item). If dropped, the stream stalls (Head-of-line blocking).
- **Unreliable**: Fire-and-forget (e.g., Footstep particles). Use for high-frequency, non-critical data.

---

## 3. The Iris Replication System (UE5 Standards)

UE5 introduced **Iris**, a high-performance replication system designed for massive player counts and optimized data-oriented processing.

### 3.1 Key Advantages of Iris
- **Internal Data Representation**: Decouples replication state from `UObject` overhead.
- **Filtering**: Advanced filtering (e.g., "Dynamic Spatial Filtering") happens at a lower level, reducing CPU cost per-client.
- **Bit-packing**: More aggressive quantization of floats and vectors.

### 3.2 Defining Replication Descriptors
Iris uses a system where the state is stripped into a "Replication Bridge." This allows the engine to compare memory blocks for changes rather than calling expensive `IsDirty()` checks on every property.

---

## 4. Client-Side Prediction & Lag Compensation

For a "AAA feel," the client cannot wait for a Round Trip Time (RTT) to see their own actions.

### 4.1 Character Movement Component (CMC) Logic
The CMC is the gold standard for prediction. 
1. **Client**: Executes move, records it in a `SavedMove` buffer, sends to server.
2. **Server**: Receives move, validates against physics.
3. **Reconciliation**: If the Server result differs from the Client's `SavedMove`, the Server sends a **Correction**. The Client "rewinds" to the error timestamp and "replays" subsequent moves.

### 4.2 Rewind Time (Lag Compensation)
To handle "favor the shooter" logic:
1. Server maintains a circular buffer of past world states (e.g., the last 200ms).
2. When a player fires, they send a timestamp.
3. Server **rewinds** the hitboxes of other players to that specific timestamp.
4. Server performs the raycast and determines the hit.

---

## 5. Network Optimization Strategies

### 5.1 Quantization (Bit-packing)
Don't send 32-bit floats for values with restricted ranges.
- **Example**: A rotation value (0-360) can be quantized into a single byte (`uint8`) if precision loss is acceptable.
- **UE5 Macro**: `UPROPERTY(Replicated, meta=(Bitmask))`

### 5.2 Actor Dormancy
If an Actor hasn't changed, set it to `DORM_Initial`. The server will stop checking its properties until a developer manually calls `ForceNetUpdate()`.

### 5.3 Net Update Frequency
Avoid the default 100Hz for non-essential actors.
- **Projectiles**: 60-100Hz.
- **Dropped Loot**: 2-5Hz.
- **Environmental Hazard**: 10Hz.

---

## 6. Mass Entity (ECS) and Networking

For thousands of entities (e.g., a massive crowd), the `AActor` overhead is too high. UE5’s **Mass Framework** uses an ECS approach.

### 6.1 Fragment Synchronization
In Mass, data is stored in **Fragments** (structs). Replication is handled via **Mass Replication Fragments**.
- **Data-Oriented**: Instead of replicating 1,000 Actors, you replicate a single "Entity Chunk" containing a contiguous array of positions.
- **Push Model**: Only fragments that have been modified are sent to the network buffer, utilizing SIMD instructions for delta-compression.

---

## 7. Security and Anti-Cheat Principles

### 7.1 Server-Side Validation
**Never trust the client.** 
- *Bad:* Client sends `ApplyDamage(100)`.
- *Good:* Client sends `Server_TryFire(Vector Location, Vector Direction)`. The Server calculates the trajectory, checks for obstacles, and applies damage.

### 7.2 Rate Limiting RPCs
Implement "Execution Budgets" for RPCs to prevent DDoS-style exploits where a client spams "UseItem" 10,000 times a second to crash the Net Driver.

---

## 8. Summary Checklist for Senior Developers

| Feature | Standard Requirement |
| :--- | :--- |
| **Movement** | Must be Predicted and Reconciled. |
| **Projectiles** | Use "Client-Side Proxy, Server-Side Simulation." |
| **UI/Inventory** | Strictly Server-Authoritative with Reliable RPCs. |
| **VFX/SFX** | Strictly Unreliable RPCs or OnRep triggers. |
| **Relevancy** | Use `NetCullDistanceSquared` for spatial optimization. |
| **Optimization** | Utilize `FArchive` bit-packing for custom structs. |

---
**Training Note:** When generating logic for multiplayer systems, always prioritize **state synchronization** (what is) over **event synchronization** (what happened). If a client joins late ("JIP" - Join In Progress), they must derive the current world state purely from replicated properties.