# AEC Project — Acoustic Echo Cancellation
**DSP Final Project | FSB Master's Program | Team of 3**

---

## Team Task Division

| Role | Owner | Scope |
|------|-------|-------|
| **Lead / Core + Desktop** | Member 1 | `core/` + `desktop-app/` |
| **Mobile (Android/IOS) ** | Member 2 | `mobile-app/android/` |
| **Integration + Docs + Testing** | Member 3 | `docs/`, E2E tests, `presentation-planning.md` |

> All three tracks run **in parallel**. Desktop ↔ Mobile LAN connection must work BEFORE AEC is integrated.

---

## Parallel Execution Strategy

```
Member 1 (Lead):        [core NLMS] → [desktop audio loop] → [LAN connect] → [plug in AEC]
Member 2 (Mobile - Android):     [Android audio capture] → [LAN socket] → [connect desktop] → [plug in AEC]
Member 3 (Integration): [presentation-planning.md] → [E2E test harness] → [metrics] → [demo script]
```

Milestone 0 (Day 1-2): Desktop ↔ Desktop raw audio over LAN — NO AEC
Milestone 1 (Day 2-3): Android ↔ Desktop raw audio over LAN
Milestone 2 (Day 3-5): Core NLMS AEC passes unit tests with synthetic echo
Milestone 3 (Day 5-6): AEC integrated into Desktop, ERLE measurable
Milestone 4 (Day 6-7): AEC integrated into Android, demo ready