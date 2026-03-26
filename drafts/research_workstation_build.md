# Axion Deep Labs — Research Workstation Build Spec
## Dual H100 NVL Deep Learning Workstation

---

## Summary

| Spec | Value |
|------|-------|
| GPU | 2x NVIDIA H100 NVL 94GB (188GB total via NVLink) |
| CPU | AMD Threadripper PRO 7975WX (32-core, 5.3GHz boost) |
| RAM | 512GB DDR5 ECC |
| Storage | 2TB + 4TB NVMe, 8TB archive |
| Total est. | ~$68,000-75,000 |

---

## Component List

### GPUs: 2x NVIDIA H100 NVL 94GB — ~$60,000

The H100 NVL is the PCIe form factor H100 with NVLink support. Two cards connected via 3 NVLink bridges give you:
- **188GB total HBM3** (pooled via NVLink) — enough for 1B+ param models in a single training run
- **600 GB/s NVLink bandwidth** (10x PCIe Gen4)
- PCIe 5.0 x16 per card, dual-slot each
- 350-400W TDP per card

**Why NVL, not standard H100 PCIe:** The standard H100 PCIe (80GB) has NO NVLink. Without NVLink, multi-GPU training communicates over PCIe, which is a massive bottleneck for model-parallel workloads. The NVL variant costs more but the 94GB memory + NVLink is what makes 1B param training viable on a workstation.

**Pricing:** Enterprise pricing is typically $27-35K per card. Request quotes from Exxact, PNY (via Amazon), or Newegg (requires compliance docs). Budget $30K/card.

**Also needed:** 3x NVLink bridge connectors (~$100-200 total)

### CPU: AMD Threadripper PRO 7975WX — ~$3,800

| Spec | Value |
|------|-------|
| Cores/Threads | 32 / 64 |
| Base / Boost | 4.0 / 5.3 GHz |
| L3 Cache | 128MB |
| TDP | 350W |
| Socket | sTR5 |
| PCIe lanes | 128x PCIe 5.0 |

**Why 32 cores, not 64 or 96:** Ripser (PH computation) is largely single-threaded — benefits from clock speed, not core count. Training data loading is parallelized but 32 cores is more than sufficient. The 7985WX (64-core, ~$7.5K) and 7995WX (96-core, ~$10K) add cost without meaningful benefit for this workload. Save $4-7K for something that matters.

### Motherboard: ASUS Pro WS TRX50-SAGE WIFI — ~$900

| Spec | Value |
|------|-------|
| Chipset | AMD TRX50 |
| Form factor | CEB (12" x 11") |
| PCIe 5.0 x16 slots | 4 (supports multi-GPU with full bandwidth) |
| RAM slots | 8x DDR5 R-DIMM (up to 1TB) |
| Networking | 10Gb + 2.5Gb LAN, WiFi 7 |
| M.2 slots | 4x PCIe 5.0 |

The standard board for Threadripper PRO multi-GPU builds. 36 power stages. Supports ECC R-DIMM. Four PCIe 5.0 x16 slots means both H100 NVLs run at full bandwidth with room to keep your 4090 in a third slot for lightweight work.

### RAM: 512GB DDR5 ECC R-DIMM — ~$1,800

- 8x 64GB DDR5-5600 ECC R-DIMM modules
- ECC is non-negotiable for multi-day training stability
- 512GB handles PH computation on large loss grids (Risk 4 scaling benchmark at 500x500 grids)
- Upgrade path to 1TB if Phase II requires larger landscape analysis

### Storage — ~$600

| Drive | Purpose | Est. Cost |
|-------|---------|-----------|
| 2TB Samsung 990 Pro (PCIe 5.0 NVMe) | OS + active experiments | ~$180 |
| 4TB WD Black SN850X (PCIe 4.0 NVMe) | Model checkpoints, loss grids | ~$280 |
| 8TB SATA SSD or HDD | Archival, dataset storage | ~$140 |

Fast NVMe for training I/O. Bulk storage for the 150+ config benchmark dataset.

### Power Supply: 1600W 80+ Titanium — ~$500

- 2x H100 NVL @ 400W = 800W
- CPU @ 350W TDP
- System draw: ~100W
- **Total peak: ~1,250W**
- 1600W gives ~20% headroom for transient spikes
- Corsair HX1500i or Seasonic PRIME TX-1600 (both Titanium rated)

### Cooling — ~$2,500

**Critical: H100 NVL cards ship with passive heatsinks.** They are designed for server chassis with high-velocity airflow. In a workstation tower, you MUST actively cool them.

**Recommended: Custom water loop covering both GPUs + CPU**

| Component | Est. Cost |
|-----------|-----------|
| 2x EK-Pro GPU WB H100 NVL water blocks | ~$800-1,000 |
| CPU water block (Threadripper compatible) | ~$150-200 |
| 360mm + 420mm radiators | ~$200-300 |
| Pump/reservoir combo | ~$200-300 |
| Fittings, tubing, coolant | ~$200-300 |
| **Total cooling** | **~$1,500-2,500** |

Water cooling also reduces the GPU from dual-slot to single-slot thickness, giving more PCIe clearance.

**Alternative:** If you'd rather not deal with custom loops, a server-style 4U rackmount chassis with high-RPM fans works. Louder, but zero maintenance. Something like a Supermicro 4U GPU chassis.

### Case — ~$300

Needs to fit:
- CEB motherboard (slightly larger than ATX)
- Dual full-length GPUs (with water blocks or passive heatsinks)
- Custom water loop radiators
- 1600W PSU

Options:
- **Fractal Design Define 7 XL** — roomy, supports CEB, excellent radiator mounting
- **Phanteks Enthoo Pro 2** — massive, supports E-ATX/CEB, good airflow
- **Lian Li O11D EVO XL** — if you want radiator-friendly layout

---

## Total Build Cost

| Component | Est. Cost |
|-----------|-----------|
| 2x H100 NVL 94GB | $60,000 |
| 3x NVLink bridges | $200 |
| AMD Threadripper PRO 7975WX | $3,800 |
| ASUS TRX50-SAGE WIFI | $900 |
| 512GB DDR5 ECC (8x64GB) | $1,800 |
| Storage (2TB + 4TB + 8TB) | $600 |
| PSU 1600W Titanium | $500 |
| Custom water cooling | $2,500 |
| Case | $300 |
| **Total** | **~$70,600** |

Add ~$2,000-3,000 contingency for cables, fans, thermal paste, misc. hardware.

**Realistic total: $72,000-75,000**

---

## What This Machine Handles

| Workload | Capability |
|----------|-----------|
| ViT-Large (307M params) training | Single GPU, room to spare |
| 1B param model training | Both GPUs via NVLink, mixed precision |
| 3B+ param model fine-tuning | Both GPUs, gradient offloading to RAM |
| PH computation (500x500 grid) | CPU + 512GB RAM, GPU idle |
| 57-config parallel runs (small models) | 4090 handles these while H100s do big work |
| Phase I full experiment set | Entire 5,000 GPU-hour plan runs locally |

---

## Cloud Break-Even

| Scenario | Cloud cost | Break-even |
|----------|-----------|------------|
| 2x H100 cloud @ $6/hr, 20 hrs/week | $12/hr = $624/week | ~2.3 years |
| 2x H100 cloud @ $6/hr, 40 hrs/week | $12/hr = $1,248/week | ~1.1 years |
| Sustained Phase I (8 hrs/day, 5 days) | $12/hr = $2,400/month | ~2.5 years |

After break-even, every hour of compute is free. The machine serves Phase I, Phase II, and any future project. Company asset on the books.

---

## SBIR Budget Line Item

In the R&D plan, this goes under **Equipment** (~$75K of ~$275K Phase I budget = 27%). NSF allows equipment purchases when justified by sustained compute need. Justification:

> Phase I requires approximately 5,000-10,000 GPU-hours across 9 months. At cloud pricing ($6/hr for dual H100), this represents $30,000-60,000 in recurring rental with no residual asset. A dedicated workstation ($75,000) reaches cost parity within Phase I and provides sustained compute capacity for Phase II development and future research programs at zero marginal cost. The machine becomes a permanent R&D asset for Axion Deep Labs.

---

## Build vs. Buy Pre-Built

| | Self-build | BIZON / Exxact pre-built |
|---|-----------|--------------------------|
| Cost | ~$72-75K | ~$85-100K (20-30% markup) |
| Warranty | Component-level (3-5 yr per part) | System-level (1-3 yr) |
| Support | Self-service | Vendor support |
| Customization | Full control | Limited to vendor configs |
| Lead time | 2-4 weeks (source parts) | 2-6 weeks (configure + ship) |
| Cooling | You build the loop | They build and test it |

**Recommendation:** Self-build if you're comfortable with custom water cooling. Go BIZON/Exxact if you want it done and tested. The $10-25K premium buys peace of mind and a single-vendor warranty.

---

## Action Items

1. **Get H100 NVL quotes:** Contact Exxact, PNY, and check Newegg (requires business compliance docs). Enterprise pricing may be below $30K/card.
2. **Decide: self-build vs. vendor.** If vendor, get a BIZON ZX4000 quote configured with 2x H100 NVL + Threadripper PRO 7975WX + 512GB ECC.
3. **Order lead time:** H100 NVL availability varies. Start sourcing GPUs first — everything else ships in days.
4. **Keep the 4090:** It becomes your dev/small-experiment machine. Don't sell it.
