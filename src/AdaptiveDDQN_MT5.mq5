
//+------------------------------------------------------------------+
//| Ilan 2.0 EA + Zones + Real DDQN (branch encoders + 2-layer fusion)                  |
//| Grid averaging with DQN + multi S/D zones + EMA/CCI/ATR          |
//| Trades CURRENT chart symbol (_Symbol)                            |
//| Quality-learning + archive-aware risk reward + stale-input cleanup        |
//|  - Persistent Q-memory + replay + target net                     |
//|  - Pending transitions for delayed outcome learning              |
//|  - DD-event memory kept as archive / training context only       |
//|  - DangerBrain retained as live protective layer                 |
//+------------------------------------------------------------------+
#property copyright "2025, CYR"
#property strict

#include <Trade/Trade.mqh>
#include <Arrays/ArrayDouble.mqh>

CTrade trade;

//  LIMITS / GLOBALS
#define MAX_SYMBOLS   16
#define REGIME_COUNT  3
#define MINI_DIM      8

int          gSymbolCount                 = 0;
string       gSymbols[MAX_SYMBOLS];
int          gMagics[MAX_SYMBOLS];
int          gPositionsCount[MAX_SYMBOLS];
CArrayDouble gTrades[MAX_SYMBOLS];
CArrayDouble gTradeLots[MAX_SYMBOLS];
datetime     gFirstTradeTime[MAX_SYMBOLS];
bool         gDeepBasketStrongEntryArmed[MAX_SYMBOLS];
datetime     gDeepBasketStrongEntryArmTime[MAX_SYMBOLS];
int          gDeepBasketStrongEntryClosedCount[MAX_SYMBOLS];

struct SDZone
{
   string   symbol;
   double   high;
   double   low;
   datetime startTime;
   datetime endTime;
   datetime breakoutTime;
   bool     isDemand;
   bool     tested;
   bool     broken;
   string   name;
};
SDZone zones[];
input int MaxZonesTracked = 50;

//--- General
input string   General              = "==== General settings ====";
input int      InpMagic             = 220111;
input double   Lots                 = 0.01;
input double   LotExponent          = 1.4;
input int      MaxTrades            = 10;
input bool     UseDeepBasketStrongFirstTrade = true;
input int      DeepBasketStrongFirstTradeMinClosed = 4;
input double   DeepBasketStrongFirstTradeMult = 2.0;
input double   ChannelCloseAlpha    = 0.30;    // close at avg +/- alpha * channelStep
input bool     UseHybridBasketTP    = true;
input int      LegacyTakeProfitPts  = 100;     // old-version style reference points
input double   BasketTPDepthFactor  = 0.20;    // deeper baskets tighten TP toward breakeven
input double   BasketTPMinFactor    = 0.15;    // never tighter than 15% of channel target
input int      Slippage             = 30;

//--- Base timeframe
input string          BaseTFSettings = "==== Base timeframe (EA logic) ====";
input ENUM_TIMEFRAMES BaseTF         = PERIOD_CURRENT;

//--- Grid (single-step BaseTF grid)
input string   GridSettings              = "==== Grid settings (single-step BaseTF) ====";

input bool     UseDynamicPips            = true;
input int      DefaultPips               = 120;      // POINTS fallback
input int      Depth                     = 24;       // bars per selected TF
input double   PipsFactor                = 3.0;      // same divisor logic
input double   GridRangeSmooth           = 0.30;     // same smoothing logic
input bool     UseStdevGridExpansion    = true;     // widen later basket adds using rolling channel-width stdev
input double   GridStdevCoeff           = 0.35;     // per-leg channel-width sigma widening from 4th position onward
input double   GridStdevMaxSigmaMult    = 2.0;      // cap total sigma widening
input int      GridStdevStartPosition   = 4;        // first position index using sigma widening
input int      GridStdevMaxPosition     = 10;       // stop increasing beyond this total basket size
input int      GridWidthStatsBars       = 100;      // rolling sample count for channel-width mean/stdev
input double   GridRollingMeanMaxWeight = 0.75;     // max pull toward rolling mean when current width is too narrow

// leg allocation
// includes first trade
// persistent DD can force extreme state
input double   ExtremeDDMoneyTrigger     = 3000.0;
input int      ExtremeDDHoursTrigger     = 2;

// chart display
input bool     ShowGridStatusPanel       = true;
input int      GridPanelCorner           = CORNER_LEFT_UPPER;
input int      GridPanelX                = 10;
input int      GridPanelY                = 20;

//--- Indicators (BaseTF)
input string   IndicatorSettings    = "==== Indicator settings (BaseTF) ====";
input int      RSI_Period           = 14;
input bool     UseCCI               = false;
input int      CCI_Period           = 55;
input int      CCI_Level            = 500;

//--- Higher TF feature packs
input string          HTFFeatures    = "==== Optional HTF features ====";
input bool            UseH1Features  = true;
input ENUM_TIMEFRAMES H1_TF          = PERIOD_H1;
input bool            UseH4Features  = true;
input ENUM_TIMEFRAMES H4_TF          = PERIOD_H4;

input int             EMA_Period     = 50;
input int             MACD_Fast      = 12;
input int             MACD_Slow      = 26;
input int             MACD_Signal    = 9;
input int             RVI_Period     = 14;

//--- ATR for regimes
input string   ATRSettings          = "==== ATR settings ====";
input int      ATR_FastPeriod       = 14;
input int      ATR_SlowPeriod       = 100;

//--- Exit / risk
input string   ExitSettings         = "==== Exit & risk settings ====";
input bool     UseTrailingStop      = false;
input int      TrailStart           = 100;
input int      TrailStop            = 100;

input bool     UseEquityStop        = false;
input double   EquityRiskPercent    = 20.0;

input bool     UseEquityLossStop        = false;
input double   EquityLossStopAmount     = 4900.0;   // money-based floating loss limit
input bool     EquityLossStopClosePositions = true;
input int      EquityLossStopCooldownSeconds = 1; // pause after forced close, then resume
// strong terminal penalty to DQN when equity stop hits
input string   GlobalWatchdogSettings   = "==== Global account watchdog ====";
input bool     UseGlobalAccountWatchdog = true;
input bool     WatchAllAccountPositions = false;   // true = include every open position on account
input int      WatchdogMagicMin         = 220100;  // used when WatchAllAccountPositions=false
input int      WatchdogMagicMax         = 220199;  // used when WatchAllAccountPositions=false


input string   ProfitPauseSettings      = "==== Profit pause after profit target ====";
input bool     UseProfitPause           = false;
input double   ProfitPauseTargetAmount  = 300.0;
input int      ProfitPauseDurationHours = 24;


//--- Virtual equity budget
input string   BudgetSettings       = "==== Virtual equity budget ====";
input double   EquityBudget         = 100000.0;
input bool     UsePerSymbolVirtualBudget = true;

//--- RL / DQN
input string   DQNSettings          = "==== DQN settings ====";
input bool     UseDQN               = true;

input int      ActionCount          = 3;      // 0 hold,1 buy,2 sell
input double   ExplorationRate      = 0.30;
input double   ExplorationDecay     = 0.998;
input double   MinExplorationRate   = 0.02;
input int      TrainingFreq         = 10;
input bool     SaveQTable           = true;

// DQN network
input string   DQNNetSettings       = "==== DQN network ====";
input int      HiddenSize           = 36;
input int      HiddenSize2          = 18;
input double   DQNLearningRate      = 0.001;
input double   DQNGamma             = 0.95;
input bool     UseInputNorm         = false;
input int      BranchEncoderMinWidth = 6;
input int      BranchEncoderH1Cap    = 64;
input int      BranchEncoderH2Cap    = 32;

// Step 1 state-architecture upgrade
input string   DDQNBranchSettings   = "==== DDQN branch / timeframe scaffold ====";
input bool     UseStateV2SelfAwareness = true;
input bool     StateNormalizationL2   = true;
input bool     UseBranchScaffold       = true;   // routing scaffold only in this step

input ENUM_TIMEFRAMES TF_EXEC         = PERIOD_CURRENT;
input ENUM_TIMEFRAMES TF_MID          = PERIOD_M15;
input ENUM_TIMEFRAMES TF_LONG         = PERIOD_H1;
input ENUM_TIMEFRAMES TF_STRUCT_EXT   = PERIOD_H4;

input bool     UseTFExecFeatures      = true;
input bool     UseTFMidFeatures       = true;
input bool     UseTFLongFeatures      = true;
input bool     UseTFStructExtFeatures = false;

input bool     UseBasketBranch        = true;
input bool     UseIndicatorBranch     = true;
input bool     UseVolatilityBranch    = true;
input bool     UseStructureBranch     = true;
input bool     UseZoneCandleBranch    = true;
input int      BasketAgeNormBars      = 288;
input double   StateDistAtrClamp      = 8.0;
input double   StateBudgetRatioClamp  = 5.0;
input int      StructLookbackExec    = 24;
input int      StructLookbackMid     = 24;
input int      StructLookbackLong    = 24;
input double   StructDistClampAtr    = 8.0;
input double   StructAmpClampAtr     = 12.0;
input int      StructHistoryBarsExec = 2880;
input int      StructHistoryBarsMid  = 960;
input int      StructHistoryBarsLong = 720;
input int      StructMaxSwingsPerTF  = 80;
input double   SwingReversalAtrMult  = 1.25;

// Double Dueling / stability controls
input bool     UseDoubleDQN         = true;
input bool     UseHuberLoss         = true;
input double   HuberDelta           = 1.0;
input bool     UseGradientClipping  = true;
input double   GradientClipValue    = 1.0;

// Multi-regime DQN bank
input string   RegimeBankSettings   = "==== Multi-regime DQN bank (3 models) ====";
input bool     UseRegimeBank        = true;
input bool     TrainAllRegimes      = true;
input bool     TrainAllRegimesWarmupOnly = true;
input int      RegimeWarmupReplayCount = 1500;
input bool     UseSoftRegimeInference = true;
input double   RegimeInferenceBlendWidth = 0.22;
input double   DangerRegimeMatchBonus = 0.08;
input double   DangerRegimeMismatchPenalty = 0.05;
input double   RegimeLowThresh      = 0.85;
input double   RegimeHighThresh     = 1.25;


// Reward architecture (risk-first DDQN)
input string   RewardSettings       = "==== DDQN reward architecture (risk-first) ====";
input double   RewardV2BasketProfitScale = 1.00;
input double   RewardV2RecoveryQualityScale = 1.00;
input double   RewardV2ProfitToMaxDDScale  = 1.10;
input double   RewardV2OneRoundBonus       = 1.50;
input double   RewardV2CleanCycleBonus     = 0.90;
input double   RewardV2CleanCycleMaxDD     = 0.02;
input bool     UseQualityAdaptiveEntry  = true;

input double   RewardV2ReversalPenaltyScale= 0.70;
input double   RewardV2OpenBaseCost        = 0.08;

input double   RewardV2AddPenaltyScale     = 0.15;
input double   RewardV2AddPenaltyQuadratic = 0.05;
input int      RewardV2DeepBasketStartPositions = 4;
input double   RewardV2DeepBasketPenaltyScale = 0.60;
input double   RewardV2AgingAddPenaltyScale = 0.12;
input double   RewardV2FailedBasketPenaltyScale = 0.65;

input double   RewardV2DDPenaltyScale      = 1.80;
input double   RewardV2EpisodeMaxDDPenaltyScale = 3.80;
input double   RewardV2DangerPenaltyScale  = 1.00;
input double   RewardV2MarginPenaltyScale  = 0.85;
input double   RewardV2AgePenaltyPerDay    = 0.14;
input double   RewardV2RewardClamp         = 10.0;

input bool     UseArchiveAwareReward            = true;
input int      ArchiveRewardScanLimit           = 600;
input double   RewardV2ArchiveOpenRiskPenaltyScale  = 0.20;
input double   RewardV2ArchiveCloseRiskPenaltyScale = 1.10;
input double   RewardV2ArchiveCleanBonusScale       = 0.45;
input double   RewardV2ArchivePatternPenaltyScale   = 0.55;
input double   RewardV2ArchiveRegimePenaltyScale    = 0.35;
input double   RewardV2RiskyProfitPenaltyScale      = 1.30;

input string   ArchiveLiveSettings                = "==== Archive live similarity / retrieval ====";
input bool     UseArchiveLiveSimilarity          = true;
input int      ArchiveLiveScanLimit              = 800;
input double   ArchiveLiveBlendWeight            = 0.18;
input double   ArchiveLiveMinBlendWeight         = 0.05;
input double   ArchiveLiveRecentHalfLifeDays     = 14.0;
input double   ArchiveLivePatternCautionScale    = 0.18;
input double   ArchiveLiveRegimeCautionScale     = 0.14;
input bool     UseArchiveAddRiskGate             = false;
input int      ArchiveAddRiskGateMinPositions    = 4;
input double   ArchiveAddRiskGateThreshold       = 2.30;
input double   ArchiveAddRiskGateCleanOffset     = 0.50;
input bool     UseEfficientPeriodLiveSupport     = true;
input double   EfficientPeriodLiveSupportScale   = 0.16;
input double   DeepSequenceAddRiskScale          = 0.16;

input string   DecisionSupportSettings           = "==== Unified live decision support ====";
input bool     UseDecisionSupportCache          = true;
input int      DecisionSupportRefreshBars       = 1;

input string   SmartAddGateSettings             = "==== Smart basket add gate ====";
input bool     UseSmartAddGate                  = true;
input double   AddGateRiskWidenThreshold        = 0.50;
input double   AddGateRiskBlockThreshold        = 0.74;
input double   AddGateSpacingMaxMult            = 2.20;
input double   AddGateLotMinScale               = 0.35;
input double   AddGateAgeSoftDays               = 1.50;

// Regime weighting
input string   RegimeWeightSettings = "==== Regime weighting ====";
input double   ExtremeRewardBoost   = 3.0;
input double   MildRewardScale      = 0.85;
input bool     SeparateExtremeStates= true;

// Training mode
input string   TrainingSettings     = "==== Training mode ====";
input bool     TrainingMode         = true;

input string   QMemorySettings      = "==== Persistent Q-memory ====";
input bool     UseQMemory           = true;
input bool     SaveQMemory          = true;
input int      MaxQMemEntries       = 30000;
input double   QMemMergeSim         = 0.97;
input double   QMemFeatureEMA       = 0.10;
input double   QMemBlendWeight      = 0.40;
input double   QMemMinConfidence    = 0.08;
input double   QMemPruneMinScore    = 0.10;

// resume behavior
input string   ResumeSettings       = "==== Resume / continuation ====";
input string   UnifiedPersistenceSettings = "==== Unified model / archive / replay persistence ====";
input bool     SaveArchiveMemory = true;
input bool     SaveReplayBanks = true;
input bool     SaveMainReplayBank = true;
input bool     SaveRecentReplayBank = true;

input bool     ResumeLowEpsilon     = true;
input double   LoadedEpsilonFactor  = 0.10;

input string   Step2Settings         = "==== Step 2 replay/target settings ====";
input bool     UseReplayBuffer       = true;
input int      ReplayCapacity        = 5000;
input int      ReplayBatchSize       = 16;
input int      ReplayWarmup          = 300;
input int      ReplayTrainIters      = 2;
input int      TargetSyncFreq        = 500;
input bool     UseTargetNet          = true;
input double   ReplayPriorityEps      = 0.01;

input string   Step11Settings         = "==== Step 11 memory/replay architecture ====";
input int      MaxEpisodeMemory       = 50000;
input int      MaxPatternMemory       = 20000;
input int      MaxRegimeEventMemory   = 10000;

input bool     UseDangerReplayBank     = true;
input bool     UseDeepBasketReplayBank = true;
input bool     UseEfficientReplayBank  = true;
input int      DangerReplayCapacity    = 30000;
input int      DeepBasketReplayCapacity = 40000;
input int      EfficientReplayCapacity  = 60000;
input int      RecentReplayCapacity     = 10000;

input int      DeepBasketAddThreshold = 3;
input double   DangerReplayDDThresholdPct = 0.05;
input double   EfficientPeriodMinRewardEfficiency = 0.50;
input int      EfficientPeriodMinTrades = 3;


input bool     UseRecentReplayBank            = true;
input bool     UseBankAwareReplaySampling      = true;
input double   ReplaySampleWeightMain          = 0.46;
input double   ReplaySampleWeightRecent        = 0.18;
input double   ReplaySampleWeightDanger        = 0.08;
input double   ReplaySampleWeightDeepBasket    = 0.08;
input double   ReplaySampleWeightEfficient     = 0.20;
input bool     UseDangerBrainReplayHook        = false;
input double   DangerReplayObsAlpha            = 0.10;
input int      EfficientPeriodWindowMinutes = 240;
input double   EfficientPeriodMaxDDPct = 0.04;
input bool     EfficientSegmentByContext      = true;
input double   EfficientSegmentMinPerMin      = 0.0010;
input double   EfficientContextBreakTolerance = 0.60;
input double   DeepSequenceEarlyBoost         = 1.60;
input double   DangerReplayEarlyStateBoost    = 1.35;
input double   DangerReplayLateRescueWeight   = 0.65;
input double   DeepReplayLateRescueWeight     = 0.55;
input double   ReplayRiskyProfitThreshold     = 1.25;
input double   ReplayMatureEfficiencyBoost    = 0.60;
input double   ReplayAntiPatternDecayScale    = 0.35;
input bool     UseReplayAwareReward           = true;
input int      ReplayRewardScanLimit          = 400;
input double   ReplayRewardHalfLifeDays       = 10.0;
input double   RewardV2ReplayOpenAntiPatternScale = 0.20;
input double   RewardV2ReplayCloseAntiPatternScale = 1.20;
input double   RewardV2ReplayEfficientBonusScale   = 0.60;
input double   RewardV2ReplayRecentCautionScale    = 0.30;
input double   RewardV2ReplayDenseAntiPatternScale = 0.35;

input string   ReplayDiagnosticsSettings      = "==== Replay diagnostics ====";
input bool     UseReplayDiagnostics           = true;
input bool     ReplayDiagnosticsToLog         = true;
input int      ReplayDiagnosticsPrintEveryBars = 24;
input bool     ReplayDiagnosticsInStatusPanel = true;

input double   DangerBankPriorityAlpha        = 0.70;
input double   DeepBankPriorityAlpha          = 0.65;
input double   EfficientBankPriorityAlpha     = 0.60;


// soft target updates
input bool     UseSoftTargetUpdate   = true;
input double   SoftTargetTau         = 0.003;

// pending transitions
input string   PendingSettings      = "==== Pending transition learning ====";
input bool     UsePendingTransitions= true;
input int      MaxPendingTransitions= 128;
input bool     UseRealPostActionTransitions = true;
input int      PendingExpireMinutes = 240;
input double   PendingOpenRewardScale = 0.10;
input double   PendingCloseRewardScale= 1.00;
input double   PendingGoodCloseBonus  = 1.0;

input string   DenseBasketLearningSettings = "==== Dense basket-health feedback ====";
input bool     UseDenseBasketHealthReward  = true;
input double   DenseBasketPnLDeltaScale    = 28.0;
input double   DenseBasketDDDeltaPenaltyScale = 6.0;
input double   DenseBasketDDRecoveryScale  = 3.0;
input double   DenseBasketAddPenaltyScale  = 1.20;
input double   DenseBasketStallAgePenaltyScale = 0.10;
input double   DenseBasketOneRoundProgressScale = 0.40;
input double   DenseBasketOpenReturnClamp  = 0.08;
input double   DenseBasketRewardClamp      = 2.5;

input string   AddSpacingGateSettings      = "==== Add spacing gate ====";
input bool     UseMinAddSpacingGate        = true;
input double   MinAddSpacingPips           = 4.0;
input double   MinAddSpacingATRFrac        = 0.08;
input double   MinAddSpacingPricePct       = 0.0002;
input bool     UseMinOpenIntervalGate      = true;
input int      MinSecondsBetweenOpens      = 30;
input int      MinExecBarsBetweenOpens     = 1;


input string   ZScoreRiskSettings      = "==== Exact StatArb z-score risk guard ====";
input bool     UseZScoreRiskGuard      = true;

// Market 1 (always available)
input string   ZScoreSymbol2           = "XAGUSD";
input ENUM_TIMEFRAMES ZScoreTF         = PERIOD_M15;
input int      ZScorePeriod            = 100;
input double   ZScoreExtremeThreshold  = 2.0;

// Optional Market 2
input bool     UseSecondZScoreMarket   = true;
input string   ZScoreSymbol2_B         = "EURUSD";
input int      ZScorePeriod_B          = 100;
input double   ZScoreExtremeThreshold_B = 2.0;

// Combination mode:
// - if UseSecondZScoreMarket=false -> Market 1 only
// - if UseSecondZScoreMarket=true and UseZScoreOrRule=true -> pause when either market is extreme
input bool     UseZScoreOrRule         = true;

input int      ZScoreResumeQuietBars   = 4;
input bool     ZScoreBlockAllTrading   = true;
input bool     UseZScoreCloseSmallBasket = true;
input int      ZScoreCloseBasketMaxTrades = 2;
input bool     UseZScoreEmergencyHedge   = true;
input int      ZScoreEmergencyHedgeMagicOffset = 700001;
input double   ZScoreEmergencyHedgeLotFactor = 1.0;
input bool     UseRecoveryAfterEmergencyHedge = true;
input int      RecoveryRestartQuietBars  = 6;
input bool     RecoveryCloseAtBreakevenOnly = true;
input double   RecoveryCombinedCloseMoney = 0.0;


input string   Step3Settings          = "==== Step 3 DD-event memory ====";
input bool     UseDDEventMemory       = true;

// soft/hard trigger on EA virtual equity DD
input double   SoftDDTriggerMoney     = 200.0;
input double   HardDDTriggerMoney     = 400.0;

// how much context to preserve around DD event
input int      BasketsBeforeDD        = 15;
input int      BasketsAfterDD         = 15;

// rolling buffers
input int      BasketHistoryCapacity  = 5000;
input int      TickTraceCapacity      = 100000;

// archive limits
input int      MaxDDEventsStored      = 3000;
input int      MaxTicksPerEvent       = 6000;
input bool     DDEventTraceOnBars     = true;
input bool     DDEventTraceUseBaseTF  = true;
input ENUM_TIMEFRAMES DDEventTraceTF  = PERIOD_M5;
input bool     SaveDDEventMemory      = true;

// phase-aware DD-event decision layer
input bool     UseDDEventBias         = true;  // bias similar historical DD contexts toward safer choices
input double   DDEventMinSim          = 0.80;
input int      DDEventMaxEventsScan   = 300;
input double   DDEventPrePenalty      = 0.25;
input double   DDEventExpandPenalty   = 0.70;
input double   DDEventRecoveryBoost   = 0.30;
input double   DDEventHoldBias        = 0.20;
input double   DDEventSameDirPenalty  = 0.20;
input double   DDEventBlendWeight     = 0.30;
input bool     DDEventUseCautionOnly  = true;

input string   DecisionBiasControlSettings = "==== DDQN primacy / subordinate bias control ====";
input double   SubordinateBiasCapFrac      = 0.50;
input double   QMemConfirmScale            = 0.85;
input double   QMemConflictToHoldScale     = 0.90;
input double   QMemConflictDirPenaltyScale = 0.55;
input double   DDEventDrawdownBoostScale   = 1.00;
input double   DangerHoldVetoScale         = 1.00;
input double   DangerDirectionalLeakScale  = 0.15;


// Zones
input string   ZoneSettings         = "==== S/D zones (BaseTF) ====";
input int      MinZoneBaseBars      = 3;
input int      MaxZoneBaseBars      = 8;
input double   ZoneMaxHeightPoints  = 300;
input int      ZoneExtendBars       = 200;
input int      MaxZonesPerBar       = 2;
input double   ZoneInvalidationBufferAtr = 0.10;
input double   ZoneMinDisplacementAtr    = 1.20;
input int      ZonePivotClusterBars      = 1;
input int      ZoneMaxLookbackSwings     = 24;
input double   ZoneNearBufferAtr         = 0.35;
input double   ZoneConfluenceAtr         = 0.50;
input double   SwingConfluenceAtr        = 0.60;

input bool     DrawSwingStructureOnChart = false;
input bool     DrawSwingZonesOnChart     = false;
input bool     DrawZoneCandleFlagsOnChart= false;

input bool     FastTrainingMode           = true;
input bool     FastTrainingSkipIndicatorBranch   = false;
input bool     FastTrainingSkipVolatilityBranch  = false;
input bool     FastTrainingSkipStructureBranch   = false;
input bool     FastTrainingSkipZoneCandleBranch  = false;
input int      ReentryCooldownSeconds    = 60;
input int      ReentryCooldownExecBars   = 2;
input bool     ForceEntryBypassesReentryCooldown = false;
input bool     UseFastBacktestStateCache  = true;
input int      SwingZoneCacheSlots        = 64;
input bool     UseSlowFeatureCaches      = true;
input bool     CacheStructureZoneOnExecBar = true;
input int      TrainEveryNDecisionBars     = 1;
input bool     FastTrainingSparseQMemory = true;
input int      FastTrainingQMemoryRefreshDecisionBars = 4;
input int      FastTrainingQMemoryRefreshMinutes      = 60;
input int      MaxDrawnSwingsPerTF       = 8;
input int      MaxDrawnZonesPerTF        = 2;


// D1 EMA trend (state)
input string   D1TrendSettings      = "==== D1 EMA trend ====";
input int      D1_EMA_Period        = 100;
input int      D1_SlopeLookbackDays = 20;
input int      D1_SideLookbackDays  = 60;
input double   D1_MaxSideDurationDays = 60.0;
// Performance / viz
input string PerformanceSettings = "==== Performance / visualization ====";
input bool   DrawZonesOnChart    = false;
input bool   VerboseLogging      = false;

input string DangerBrainSettings  = "==== Danger Brain (Regime + Memory) ====";
input bool   UseDangerBrain       = true;

input ENUM_TIMEFRAMES FingerprintTF   = PERIOD_CURRENT;
input int    FingerprintBars          = 32;

// thresholds are copied into runtime vars (so we can calibrate)
input double T1_CautionEnter          = 0.55;
input double T1_CautionExit           = 0.45;
input double T2_DangerEnter           = 0.75;
input double T2_DangerExit            = 0.65;
input int    ModeCooldownMinutes      = 60;

input double LRScale_Caution          = 0.30;
input double LRScale_Danger           = 0.10;

input bool   DangerBlocksNewEntries   = false;
input double ScoreW_SimContrast       = 1.0;
input double ScoreW_Exposure          = 0.6;

input double ExposureW_Positions      = 0.10;
input double ExposureW_DD             = 2.0;

input bool   SaveDangerMemory         = true;

input string DangerSmartnessSettings  = "==== Danger smartness pack ====";

input bool   UseDeltaSimilarity       = true;
input double SimW_DeltaFingerprint    = 0.70;
input double SimW_DeltaMiniState      = 0.50;

// 0=original block rules, 1=danger hold only, 2=danger with-trend only
input int    DangerActionPolicy       = 0;

input string BadEpisodeSettings        = "==== Bad Episode Learning ====";
input double BadProtoMergeSim          = 0.96;
input double BadProtoFeatureEMA        = 0.20;
input double BadProtoBiasEMA           = 0.20;

input double BadBias_HoldBoost         = 0.70;
input double BadBias_AgainstPenalty    = 0.70;
input double BadBias_WithTrendBoost    = 0.25;
input double BadBias_RepeatPenalty     = 0.35;

input string SafeRecoverySettings      = "==== Safe Recovery Learning ====";
input string MemoryAgingSettings       = "==== Memory Aging / Pruning ====";
input double ProtoHalfLifeDays         = 360.0;
input int    MaxProtosStored           = 12000;
input double PruneMinKeepScore         = 0.1;
input bool   PruneOnDeinit             = true;

input string MiniStateSettings        = "==== Mini State Fingerprint ====";
input bool   UseMiniStateFingerprint  = true;

input double SimW_PriceFingerprint    = 1.0;
input double SimW_MiniState           = 0.8;

// indices based on BuildState push order
int gMiniIdx[MINI_DIM] = { 2, 6, 7, 1, 11, 12, 14, 15 };

input string ProfitRewardSettings      = "==== Profit reward shaping ====";
input double ProfitReturnClamp         = 0.80;
input double ProfitRewardScale         = 10.0;

input string RecoveryQualitySettings   = "==== Recovery quality shaping ====";
input double RecWinRatioScale          = 2.0;
input double RecLossCountPenalty       = 1.0;
input double RecTradesUsedPenalty      = 0.40;
input double RecQualityClamp           = 8.0;

input string PositionLearningSettings  = "==== Per-position basket learning ====";
input bool   UsePerPositionCloseLearning = true;
input double LegPnLPerLotScale         = 0.05;
input double LegDistancePointsScale    = 0.002;
input double LegIndexRewardScale       = 0.03;
input double LegHoldHoursPenaltyScale  = 0.01;
input int    MaxClosedLegHistory       = 3000;

input string   ForcedEntrySettings       = "==== Forced entry watchdog ====";
input bool     UseForcedEntryWatchdog    = true;
input int      ForcedEntryIdleMinutes    = 60;
input int      ForcedEntryWindowMinutes  = 15;
input bool     ForcedEntryIgnoreDanger   = true;

datetime gLastTradeOpenTime[MAX_SYMBOLS];
bool     gForcedEntryActive[MAX_SYMBOLS];
datetime gForcedEntryArmTime[MAX_SYMBOLS];
datetime gForcedEntryDeadline[MAX_SYMBOLS];

double   maxEquity             = 0.0;
double   gTickEquityBaseline   = 0.0;
double   gTickBalanceBaseline  = 0.0;
datetime gRewardBaselineTick   = 0;
double   gEAStartEquity        = 0.0;
double   gEAClosedProfit       = 0.0;

datetime gEquityLossStopResumeTime = 0;
datetime gProfitPauseResumeTime    = 0;
double   gProfitCycleClosedProfit  = 0.0;


double   gLastForcedStopLossMoney[MAX_SYMBOLS];
datetime gLastForcedStopTime[MAX_SYMBOLS];
bool     gLastForcedStopPendingLearn[MAX_SYMBOLS];

double   gForcedStopFP[MAX_SYMBOLS][6];
bool     gForcedStopHaveFP[MAX_SYMBOLS];

double   gForcedStopMini[MAX_SYMBOLS][MINI_DIM];
bool     gForcedStopHaveMini[MAX_SYMBOLS];

double   gForcedStopQ[MAX_SYMBOLS][8];
bool     gForcedStopHaveQ[MAX_SYMBOLS];

int      gForcedStopBasketDir[MAX_SYMBOLS];
int      gForcedStopTrendDir[MAX_SYMBOLS];
double   gForcedStopAtrRatio[MAX_SYMBOLS];
int      gForcedStopLabel[MAX_SYMBOLS];

// RL/runtime caches for efficiency
int      tickCounter          = 0;
double   currentEpsilon       = 0.0;
bool     isTraining           = true;
double   totalReward          = 0.0;
int      episodeCount         = 0;
int      gTargetSyncCounter   = 0;
double   gAccountEquityStopPeak = 0.0;

datetime gZScoreLastClosedBarTF[MAX_SYMBOLS];

// Effective / combined z-score state used by risk management and DDQN
double   gZScoreLastValue[MAX_SYMBOLS];
double   gZScoreLastAbs[MAX_SYMBOLS];
bool     gZScoreExtremeNow[MAX_SYMBOLS];
bool     gZScorePauseTrading[MAX_SYMBOLS];
int      gZScoreQuietBars[MAX_SYMBOLS];
datetime gZScoreLastExtremeBarTime[MAX_SYMBOLS];
datetime gZScoreCloseHandledBar[MAX_SYMBOLS];
bool     gZScoreEmergencyHedgeActive[MAX_SYMBOLS];
int      gZScoreEmergencyOriginalDir[MAX_SYMBOLS];
datetime gZScoreEmergencyHedgeBar[MAX_SYMBOLS];

// Per-market monitoring state
double   gZScoreLastValueA[MAX_SYMBOLS];
double   gZScoreLastAbsA[MAX_SYMBOLS];
bool     gZScoreExtremeA[MAX_SYMBOLS];
double   gZScoreLastValueB[MAX_SYMBOLS];
double   gZScoreLastAbsB[MAX_SYMBOLS];
bool     gZScoreExtremeB[MAX_SYMBOLS];


double   gCachedSymbolPnL[MAX_SYMBOLS];
bool     gHasPositionTypeCache[MAX_SYMBOLS];
int      gBasketDirCache[MAX_SYMBOLS];
double   gBasketAvgPriceCache[MAX_SYMBOLS];
bool     gBasketAvgValid[MAX_SYMBOLS];

struct DQNNetwork
{
   int    input_dim;
   int    hidden_dim;
   int    hidden_dim2;
   int    output_dim;
   int    fusion_dim;

   int    basket_h1;
   int    basket_h2;
   int    indicator_h1;
   int    indicator_h2;
   int    volatility_h1;
   int    volatility_h2;
   int    structure_h1;
   int    structure_h2;
   int    zone_h1;
   int    zone_h2;

   // branch encoders (2 dense layers each)
   double basket_W1[];
   double basket_b1[];
   double basket_W2[];
   double basket_b2[];

   double indicator_W1[];
   double indicator_b1[];
   double indicator_W2[];
   double indicator_b2[];

   double volatility_W1[];
   double volatility_b1[];
   double volatility_W2[];
   double volatility_b2[];

   double structure_W1[];
   double structure_b1[];
   double structure_W2[];
   double structure_b2[];

   double zone_W1[];
   double zone_b1[];
   double zone_W2[];
   double zone_b2[];

   // shared fusion trunk
   double W1[];
   double b1[];
   double W2[];
   double b2[];

   // dueling heads
   double WV[];
   double bV[];

   double WA[];
   double bA[];

   double feat_mean[];
   double feat_std[];
};

DQNNetwork gDQN[MAX_SYMBOLS][REGIME_COUNT];
DQNNetwork gTargetDQN[MAX_SYMBOLS][REGIME_COUNT];

int W1Index(const int input_dim, int h, int i)    { return h*input_dim + i; }
int WVIndex(const int hidden_dim, int v, int h)   { return v*hidden_dim + h; }
int WAIndex(const int hidden_dim, int o, int h)   { return o*hidden_dim + h; }

int DenseIndex(const int input_dim, int row, int col) { return row*input_dim + col; }

int HeadInputDim(const DQNNetwork &net)
{
   return (net.hidden_dim2>0 ? net.hidden_dim2 : net.hidden_dim);
}

ENUM_TIMEFRAMES DDEventTraceTFForSymbol(const string symbol)
{
   if(DDEventTraceUseBaseTF)
      return BaseTF;
   if(DDEventTraceTF!=PERIOD_CURRENT)
      return DDEventTraceTF;
   return BaseTF;
}


int BranchHidden1Size(const int inputCount)
{
   if(inputCount<=0) return 0;
   int minW=MathMax(2, BranchEncoderMinWidth);
   int cap=MathMax(minW, BranchEncoderH1Cap);
   int v=MathMax(minW, inputCount*2);
   return MathMin(v, cap);
}

int BranchHidden2Size(const int inputCount,const int h1)
{
   if(inputCount<=0 || h1<=0) return 0;
   int minW=MathMax(2, BranchEncoderMinWidth);
   int cap=MathMax(minW, BranchEncoderH2Cap);
   int v=MathMax(minW, inputCount);
   v=MathMin(v, cap);
   return MathMin(v, h1);
}

void InitBranchEncoder(const int inputCount,
                       const int h1,
                       const int h2,
                       double &W1[],
                       double &b1[],
                       double &W2[],
                       double &b2[])
{
   ArrayResize(W1, MathMax(inputCount,0)*MathMax(h1,0));
   ArrayResize(b1, MathMax(h1,0));
   ArrayResize(W2, MathMax(h1,0)*MathMax(h2,0));
   ArrayResize(b2, MathMax(h2,0));

   if(inputCount<=0 || h1<=0 || h2<=0)
      return;

   double scale1 = 1.0 / MathSqrt((double)MathMax(inputCount,1));
   double scale2 = 1.0 / MathSqrt((double)MathMax(h1,1));

   for(int i=0;i<ArraySize(W1);i++)
   {
      double r=(double)MathRand()/32767.0;
      W1[i]=(r*2.0-1.0)*scale1;
   }
   for(int i=0;i<ArraySize(b1);i++)
      b1[i]=0.0;

   for(int i=0;i<ArraySize(W2);i++)
   {
      double r=(double)MathRand()/32767.0;
      W2[i]=(r*2.0-1.0)*scale2;
   }
   for(int i=0;i<ArraySize(b2);i++)
      b2[i]=0.0;
}

void ForwardBranchEncoder(const double &x[],
                          const int start,
                          const int inputCount,
                          const double &W1[],
                          const double &b1[],
                          const double &W2[],
                          const double &b2[],
                          double &z1[],
                          double &a1[],
                          double &z2[],
                          double &a2[])
{
   int h1=ArraySize(b1);
   int h2=ArraySize(b2);

   ArrayResize(z1,h1);
   ArrayResize(a1,h1);
   ArrayResize(z2,h2);
   ArrayResize(a2,h2);

   if(inputCount<=0 || h1<=0 || h2<=0)
      return;

   int nx=ArraySize(x);

   for(int r=0;r<h1;r++)
   {
      double sum=b1[r];
      for(int c=0;c<inputCount;c++)
      {
         int xi=start+c;
         double xv=(xi>=0 && xi<nx ? x[xi] : 0.0);
         sum += W1[DenseIndex(inputCount,r,c)] * xv;
      }
      z1[r]=sum;
      a1[r]=SiLU(sum);
   }

   for(int r=0;r<h2;r++)
   {
      double sum=b2[r];
      for(int c=0;c<h1;c++)
         sum += W2[DenseIndex(h1,r,c)] * a1[c];
      z2[r]=sum;
      a2[r]=SiLU(sum);
   }
}

void BackpropBranchEncoder(const double &x[],
                           const int start,
                           const int inputCount,
                           double &W1[],
                           double &b1[],
                           double &W2[],
                           double &b2[],
                           const double &z1[],
                           const double &a1[],
                           const double &z2[],
                           double &gradOut[],
                           const double lr)
{
   int h1=ArraySize(b1);
   int h2=ArraySize(b2);
   if(inputCount<=0 || h1<=0 || h2<=0) return;

   double dZ2[];
   ArrayResize(dZ2,h2);
   for(int i=0;i<h2;i++)
      dZ2[i]=gradOut[i]*SiLUDerivativeFromPreAct(z2[i]);
   if(UseGradientClipping) ClipArrayInPlace(dZ2,GradientClipValue);

   double oldW2[];
   ArrayResize(oldW2,ArraySize(W2));
   for(int i=0;i<ArraySize(W2);i++) oldW2[i]=W2[i];

   for(int r=0;r<h2;r++)
   {
      for(int c=0;c<h1;c++)
      {
         int idx=DenseIndex(h1,r,c);
         double grad=dZ2[r]*a1[c];
         if(UseGradientClipping) grad=ClipScalar(grad,GradientClipValue);
         W2[idx] -= lr*grad;
      }
      double gradb=dZ2[r];
      if(UseGradientClipping) gradb=ClipScalar(gradb,GradientClipValue);
      b2[r] -= lr*gradb;
   }

   double dA1[];
   ArrayResize(dA1,h1);
   for(int c=0;c<h1;c++)
   {
      double s=0.0;
      for(int r=0;r<h2;r++)
         s += oldW2[DenseIndex(h1,r,c)] * dZ2[r];
      dA1[c]=s;
   }

   double dZ1[];
   ArrayResize(dZ1,h1);
   for(int i=0;i<h1;i++)
      dZ1[i]=dA1[i]*SiLUDerivativeFromPreAct(z1[i]);
   if(UseGradientClipping) ClipArrayInPlace(dZ1,GradientClipValue);

   int nx=ArraySize(x);
   for(int r=0;r<h1;r++)
   {
      for(int c=0;c<inputCount;c++)
      {
         int xi=start+c;
         double xv=(xi>=0 && xi<nx ? x[xi] : 0.0);
         int idx=DenseIndex(inputCount,r,c);
         double grad=dZ1[r]*xv;
         if(UseGradientClipping) grad=ClipScalar(grad,GradientClipValue);
         W1[idx] -= lr*grad;
      }
      double gradb=dZ1[r];
      if(UseGradientClipping) gradb=ClipScalar(gradb,GradientClipValue);
      b1[r] -= lr*gradb;
   }
}


struct QMemEntry
{
   int      regime;
   datetime created;
   datetime lastUsed;
   uint     usedCount;

   double   stateKey[];
   double   qVals[];
   double   conf;
   double   score;
};

QMemEntry gQMem[];


struct ReplayItem
{
   int      symIdx;
   int      regime;
   int      action;
   double   reward;
   bool     done;
   double   state[];
   double   nextState[];
   double   priority;

   int      replayBankType;
   long     episodeId;
   long     patternId;
   long     regimeId;
   long     periodId;

   int      basketStateClass;
   int      addDepthClass;
   int      dangerClass;
   int      zoneContextClass;
   int      structureContextClass;
   int      liquidityClass;

   datetime eventTime;
   double   painSeverity;
   double   recurrenceScore;
   double   regimeBreakScore;
   double   macroSig[];
   double   microSig[];
};

enum ReplayBankType
{
   REPLAY_BANK_RECENT = 0,
   REPLAY_BANK_DANGER = 1,
   REPLAY_BANK_DEEP_BASKET = 2,
   REPLAY_BANK_EFFICIENT = 3
};

enum MemoryZoneContextClass
{
   MEM_ZONE_NONE = 0,
   MEM_ZONE_NEAR_DEMAND = 1,
   MEM_ZONE_NEAR_SUPPLY = 2,
   MEM_ZONE_INSIDE_DEMAND = 3,
   MEM_ZONE_INSIDE_SUPPLY = 4
};

enum MemoryStructureContextClass
{
   MEM_STRUCT_MIXED = 0,
   MEM_STRUCT_TREND_UP = 1,
   MEM_STRUCT_TREND_DOWN = 2,
   MEM_STRUCT_COMPRESSION = 3,
   MEM_STRUCT_EXPANSION = 4
};

struct BarSnapshot
{
   datetime time;
   ENUM_TIMEFRAMES tf;

   double open;
   double high;
   double low;
   double close;
   long   tickVolume;
   int    spreadPoints;

   double atr;
   double realizedVol;
   double adx;
   double rsi;
   double priceVsEMA;

   double structTendency;
   double swingHighDistAtr;
   double swingLowDistAtr;
   double zoneRelevance;
   double rejectionStrength;
   double acceptanceStrength;
   double indecisionState;
};

struct BarSpanRef
{
   ENUM_TIMEFRAMES tf;
   datetime startTime;
   datetime endTime;
   int startIndex;
   int endIndex;
};

struct EpisodeMemory
{
   long   episodeId;
   string symbol;
   int    symIdx;

   datetime startTime;
   datetime endTime;

   int basketDir;
   int openPositionsMax;
   int addCount;
   int actionsCount;

   double entryPriceFirst;
   double avgEntryAtWorst;
   double closePriceFinal;

   double pnlFinal;
   double rewardTotal;
   double rewardEfficiency;
   double maxDrawdownPct;
   double maxDangerScore;
   double maxMarginStress;

   int oneRoundTrade;
   int forcedStopLikeEvent;
   int inefficientRecovery;

   int sessionType;
   int liquidityType;
   int regimeType;
   int patternType;

   BarSpanRef execSpan;
   BarSpanRef midSpan;
   BarSpanRef longSpan;
   BarSpanRef structExtSpan;
};

struct PatternMemory
{
   long patternId;
   string symbol;
   int symIdx;

   int patternType;
   int strengthClass;
   int volatilityClass;
   int liquidityClass;

   datetime startTime;
   datetime endTime;

   double rewardEfficiencyMean;
   double ddMean;
   double addMean;

   BarSpanRef execSpan;
   BarSpanRef midSpan;
   BarSpanRef longSpan;

   long linkedEpisodeIds[];
};

struct RegimeEventMemory
{
   long regimeId;
   string symbol;
   int symIdx;

   int regimeType;
   int eventType;

   datetime startTime;
   datetime endTime;

   double avgVol;
   double avgSpread;
   double avgADX;
   double avgRewardEfficiency;

   BarSpanRef execSpan;
   BarSpanRef midSpan;
   BarSpanRef longSpan;

   long linkedEpisodeIds[];
   long linkedPatternIds[];
};

struct BasketSequenceRef
{
   long episodeId;
   int symIdx;

   datetime startTime;
   datetime endTime;

   int basketDir;
   int addCount;
   int maxPositions;
   double maxDD;
   double finalReward;

   int replayItemIndexes[];
};

struct EfficientPeriodRef
{
   long periodId;
   int symIdx;

   datetime startTime;
   datetime endTime;

   double rewardTotal;
   double rewardEfficiency;
   double ddMax;
   int oneRoundCount;
   int addCountTotal;

   int sessionType;
   int liquidityType;
   int regimeType;
   int patternType;

   int replayItemIndexes[];
};

struct ReplayBankStore
{
   ReplayItem items[];
   int count;
   int maxCount;
};

struct DeepBasketReplayStore
{
   ReplayItem items[];
   BasketSequenceRef sequences[];
   int itemCount;
   int seqCount;
   int maxItems;
   int maxSeqs;
};

struct EfficientReplayStore
{
   ReplayItem items[];
   EfficientPeriodRef periods[];
   int itemCount;
   int periodCount;
   int maxItems;
   int maxPeriods;
};

EpisodeMemory         gEpisodeMemory[];
PatternMemory         gPatternMemory[];
RegimeEventMemory     gRegimeEventMemory[];

ReplayBankStore       gDangerReplayBank;
DeepBasketReplayStore gDeepBasketReplayBank;
EfficientReplayStore  gEfficientReplayBank;
ReplayBankStore       gRecentReplayBank;

double                gDangerReplaySeverityEMA[MAX_SYMBOLS];
int                   gDangerReplayHits[MAX_SYMBOLS];
double                gDeepBasketReplaySeverityEMA[MAX_SYMBOLS];
int                   gDeepBasketReplayHits[MAX_SYMBOLS];
int                   gReplayDiagWindowRecentAdds[MAX_SYMBOLS];
int                   gReplayDiagWindowDangerAdds[MAX_SYMBOLS];
int                   gReplayDiagWindowDeepAdds[MAX_SYMBOLS];
int                   gReplayDiagWindowEfficientAdds[MAX_SYMBOLS];
int                   gReplayDiagBarsSincePrint[MAX_SYMBOLS];

struct ActiveBasketEpisode
{
   bool     active;
   long     episodeId;
   int      symIdx;
   int      basketDir;
   datetime startTime;
   datetime lastTime;
   int      addCount;
   int      maxPositions;
   double   maxDD;
   double   rewardAccum;
   int      replayItemIndexes[];
};

struct ActiveEfficientPeriod
{
   bool     active;
   long     periodId;
   int      symIdx;
   datetime startTime;
   datetime lastTime;
   double   rewardTotal;
   double   ddMax;
   int      oneRoundCount;
   int      addCountTotal;
   int      tradeCount;
   int      lastLiquidityClass;
   int      lastRegime;
   int      lastStructureClass;
   int      contextBreakCount;
   int      replayItemIndexes[];
};

ActiveBasketEpisode   gActiveBasketEpisodes[MAX_SYMBOLS];
ActiveEfficientPeriod gActiveEfficientPeriods[MAX_SYMBOLS];
long                  gNextEpisodeId = 1;
long                  gNextPeriodId  = 1;

ReplayItem gReplay[];

struct PendingTransition
{
   bool     active;
   int      symIdx;
   int      regime;
   int      action;
   datetime created;
   int      basketDir;
   int      positionsAtOpen;
   int      legIndex;
   datetime entryTime;
   datetime closeTime;
   double   entryPrice;
   double   closePrice;
   double   entryVolume;
   double   basketAvgAtEntry;
   double   individualPnL;
   double   denseRewardAccum;
   long     episodeId;
   double   state[];
};

PendingTransition gPending[];

struct DenseBasketHealthTracker
{
   bool     active;
   datetime lastBarTime;
   double   prevOpenReturn;
   double   prevDD;
   int      prevPositions;
   double   prevAgeDays;
};

DenseBasketHealthTracker gDenseBasketHealth[MAX_SYMBOLS];

struct PositionCloseItem
{
   ulong    ticket;
   int      basketDir;
   int      legIndex;
   datetime entryTime;
   datetime closeTime;
   double   entryPrice;
   double   closePrice;
   double   volume;
   double   profit;
};

struct BasketCloseResult
{
   int      attempted;
   int      closed;
   double   attemptedVolume;
   double   closedVolume;
   double   attemptedProfit;
   double   closedProfit;
   int      wins;
   int      losses;
   int      total;
   bool     allClosed;
};

struct ClosedLegLearningRecord
{
   datetime entryTime;
   datetime closeTime;
   string   symbol;
   int      regime;
   int      action;
   int      basketDir;
   int      legIndex;
   double   entryPrice;
   double   closePrice;
   double   entryVolume;
   double   basketAvgAtEntry;
   double   individualPnL;
   double   resolvedReward;
};

ClosedLegLearningRecord gClosedLegHistory[];

struct BasketSnapshot
{
   datetime timeStamp;
   string   symbol;
   int      magic;
   int      regime;
   int      basketDir;
   int      positionsCount;
   double   equity;
   double   openPnL;
   double   avgPrice;
   double   midPrice;
   double   entryPrice;
   double   gridStep;
   double   danger;
   double   stateKey[];
   double   qVals[];
};

struct TickTraceItem
{
   datetime timeStamp;
   double   bid;
   double   ask;
   double   mid;
   double   spreadPoints;
   double   equity;
   double   danger;
};

struct DDEventRecord
{
   datetime created;
   datetime triggerTime;
   string   symbol;
   int      magic;
   int      regimeAtTrigger;
   int      basketDirAtTrigger;

   double   ddAtTrigger;
   bool     hardTrigger;
   bool     completed;

   BasketSnapshot preBaskets[];
   BasketSnapshot postBaskets[];
   TickTraceItem  ticks[];

   double   triggerStateKey[];
   double   triggerQVals[];
   double   peakDD;
   int      basketDepthMax;
   double   timeUnderWaterNorm;
   double   recoveryFailureScore;
   double   painSeverity;
   int      eventType;
   double   macroSig[];
   double   microSig[];
};

BasketSnapshot gBasketHistory[];
TickTraceItem  gTickTrace[];
DDEventRecord  gDDEvents[];

bool     gDDEventActive            = false;
datetime gDDEventTriggerTime       = 0;
double   gDDEventTriggerDD         = 0.0;
bool     gDDEventHard              = false;
int      gDDEventStartBasketIndex  = -1;
int      gDDEventStartTickIndex    = -1;
int      gDDEventPostBasketCount   = 0;
datetime gDDEventLastTraceBarTime[MAX_SYMBOLS];

enum BrainMode { MODE_NORMAL=0, MODE_CAUTION=1, MODE_DANGER=2 };
enum EpisodeLabel { LBL_AGAINST_UPTREND=0, LBL_AGAINST_DOWNTREND=1 };

struct ProtoEntry
{
   int      label;
   int      basketDir;
   int      trendDir;
   double   atrRatio;
   datetime created;

   bool     isDanger;
   double   features[];
   double   stateMini[];
   double   deltaFP[];
   double   deltaMini[];

   double   adapterBias[];
   double   qSnap[];
   double   painMean;
   double   painCount;
   double   macroSig[];
   double   microSig[];
   double   deepBasketRate;
   double   regimeBreakRate;
   double   counterTrendFailureRate;
   double   reversalTrapRate;
   double   recoveryFailureRate;
   double   survivalScore;
   uint     usedCount;
   datetime lastUsed;
};

BrainMode gMode[MAX_SYMBOLS];
datetime  gModeLastChange[MAX_SYMBOLS];
double    gPDanger[MAX_SYMBOLS];
double    gLRScale[MAX_SYMBOLS];

double gT1Enter=0.0, gT1Exit=0.0, gT2Enter=0.0, gT2Exit=0.0;

int gCalibFalseDanger=0;
int gCalibMissedDanger=0;
int gCalibTrueDanger=0;

datetime gFPLastBar[MAX_SYMBOLS];
double   gFPCache[MAX_SYMBOLS][6];
double   gFPPrev[MAX_SYMBOLS][6];
bool     gFPPrevValid[MAX_SYMBOLS];

datetime gMiniLastBar[MAX_SYMBOLS];
double   gMiniCache[MAX_SYMBOLS][MINI_DIM];
bool     gMiniValid[MAX_SYMBOLS];
double   gMiniPrev[MAX_SYMBOLS][MINI_DIM];
bool     gMiniPrevValid[MAX_SYMBOLS];

ProtoEntry gProtos[];

struct DecisionSupportContext
{
   bool     valid;
   bool     forcedEntry;
   datetime barTime;
   datetime refreshTime;
   int      regime;
   int      positionsCount;
   int      basketDir;
   int      brainMode;

   double   actionDelta[3];
   double   qMemoryAgreement;
   double   qMemoryConflict;
   double   archiveCaution;
   double   ddEventRisk;
   double   dangerProbability;
   double   oneRoundPrior;
   double   addRiskPrior;
   double   supportConfidence;
   double   painRecurrenceRisk;
   double   macroReversalTrapPrior;
   double   counterTrendFailurePrior;
   double   regimeBreakPainPrior;
   double   recoveryFalseStartRisk;
   double   deepBasketPainPrior;
   double   painMemoryAgreement;
   double   macroMicroConflict;
   double   lateTrendFadePenalty;
   double   painConfidence;
   double   trendPersistenceProb;
   double   trendReversalProb;
   double   spikeRiskProb;
   double   expectedBasketDepth;
   double   trendContinuationQuality;
   double   breakoutReclaimQuality;
   double   reversalTransitionQuality;
   double   modeDominanceScore;
   double   modeConflictScore;
};

DecisionSupportContext gDecisionSupportCache[MAX_SYMBOLS];

datetime gBadDDStart[MAX_SYMBOLS];
bool     gBadDDConfirmed[MAX_SYMBOLS];
int      gBadLabel[MAX_SYMBOLS];
int      gBadBasketDir[MAX_SYMBOLS];
int      gBadTrendDir[MAX_SYMBOLS];
double   gBadAtrRatio[MAX_SYMBOLS];

double   gBadFPStart[MAX_SYMBOLS][6];
double   gBadFPBasketOpen[MAX_SYMBOLS][6];
bool     gBadHaveBasketFP[MAX_SYMBOLS];

double   gBadQStart[MAX_SYMBOLS][8];
bool     gBadHaveQStart[MAX_SYMBOLS];

double   gBadFPPre[MAX_SYMBOLS][6];
bool     gBadHavePreFP[MAX_SYMBOLS];

double   gBadFPEnd[MAX_SYMBOLS][6];
bool     gBadHaveEndFP[MAX_SYMBOLS];

double   gBadMiniStart[MAX_SYMBOLS][MINI_DIM];
double   gBadMiniBasketOpen[MAX_SYMBOLS][MINI_DIM];
bool     gBadHaveMiniStart[MAX_SYMBOLS];
bool     gBadHaveMiniBasket[MAX_SYMBOLS];
double   gBadMiniPre[MAX_SYMBOLS][MINI_DIM];
bool     gBadHaveMiniPre[MAX_SYMBOLS];
double   gBadMiniEnd[MAX_SYMBOLS][MINI_DIM];
bool     gBadHaveMiniEnd[MAX_SYMBOLS];

datetime gLastBarTime[MAX_SYMBOLS];
datetime gLastReplayDecisionBarTime[MAX_SYMBOLS];
int      gReplayDecisionBarCounter[MAX_SYMBOLS];
int      gReplayPendingTrainCount=0;
bool     gLoadedDQNFastModeMeta[MAX_SYMBOLS];
string   gLoadedDQNMetaNote[MAX_SYMBOLS];

datetime gDecisionCtxBarTime[MAX_SYMBOLS];
int      gDecisionCtxHourBucket[MAX_SYMBOLS];
int      gDecisionCtxRegimeType[MAX_SYMBOLS];
int      gDecisionCtxPatternType[MAX_SYMBOLS];
int      gDecisionCtxLiquidityType[MAX_SYMBOLS];
int      gDecisionCtxSessionType[MAX_SYMBOLS];
double   gDecisionCtxAtrRatio[MAX_SYMBOLS];
bool     gDecisionCtxValid[MAX_SYMBOLS];


datetime gSignalTrackBarTime[MAX_SYMBOLS];
int      gSignalTrackRegime[MAX_SYMBOLS];
int      gSignalTrackDirCandidate[MAX_SYMBOLS];
int      gSignalTrackPersistCount[MAX_SYMBOLS];
int      gSignalTrackMode[MAX_SYMBOLS];
double   gSignalTrackMacroBias[MAX_SYMBOLS];
double   gSignalTrackTransition[MAX_SYMBOLS];

#define DIR_STATE_DOWN   -1
#define DIR_STATE_NEUTRAL 0
#define DIR_STATE_UP      1


datetime gFastQMemLastRefreshTime[MAX_SYMBOLS];
datetime gFastQMemLastDecisionBarTime[MAX_SYMBOLS];
int      gFastQMemDecisionBarCounter[MAX_SYMBOLS];
int      gFastQMemCachedRegime[MAX_SYMBOLS];
bool     gFastQMemCachedValid[MAX_SYMBOLS];
double   gFastQMemCachedConf[MAX_SYMBOLS];
double   gFastQMemCachedQ[MAX_SYMBOLS][3];

datetime gD1LastBarTime[MAX_SYMBOLS];
double   gD1DistCache[MAX_SYMBOLS];
double   gD1SlopeCache[MAX_SYMBOLS];
double   gD1SideDurCache[MAX_SYMBOLS];

datetime gATRLastBarTime[MAX_SYMBOLS];
double   gATRratioCache[MAX_SYMBOLS];
datetime gGridLastBar[MAX_SYMBOLS];
double   gGridStepCache[MAX_SYMBOLS];
double   gGridWidthSigmaCache[MAX_SYMBOLS];

// kept only for compatibility with panel / snapshot code
int      gGridActiveChannel[MAX_SYMBOLS];
double   gGridActiveStep[MAX_SYMBOLS];

datetime gExtremeDDStart[MAX_SYMBOLS];
bool     gExtremeDDArmed[MAX_SYMBOLS];

string   gGridPanelName = "DQN_GRID_STATUS_PANEL";

// BaseTF indicator handles + caches
int      hRSI_Base[MAX_SYMBOLS];
int      hCCI_Base[MAX_SYMBOLS];
int      hMACD_Base[MAX_SYMBOLS];
int      hEMA_Base[MAX_SYMBOLS];
int      hRVI_Base[MAX_SYMBOLS];
int      hATRfast_Base[MAX_SYMBOLS];
int      hATRslow_Base[MAX_SYMBOLS];
datetime gIndLastBarTime[MAX_SYMBOLS];

double   gRSI_Base[MAX_SYMBOLS];
double   gCCI_Base[MAX_SYMBOLS];
double   gMACD_BaseMain[MAX_SYMBOLS];
double   gEMA_BaseVal[MAX_SYMBOLS];
double   gRVI_BaseMain[MAX_SYMBOLS];
double   gATRfast_BaseVal[MAX_SYMBOLS];
double   gATRslow_BaseVal[MAX_SYMBOLS];

// H1 handles + caches
int      hRSI_H1[MAX_SYMBOLS];
int      hMACD_H1[MAX_SYMBOLS];
int      hEMA_H1[MAX_SYMBOLS];
int      hRVI_H1[MAX_SYMBOLS];
int      hATRfast_H1[MAX_SYMBOLS];
int      hATRslow_H1[MAX_SYMBOLS];
datetime gH1LastBar[MAX_SYMBOLS];

double   gRSI_H1v[MAX_SYMBOLS];
double   gMACD_H1v[MAX_SYMBOLS];
double   gEMA_H1v[MAX_SYMBOLS];
double   gRVI_H1v[MAX_SYMBOLS];
double   gATRratio_H1[MAX_SYMBOLS];

// H4 handles + caches
int      hRSI_H4[MAX_SYMBOLS];
int      hMACD_H4[MAX_SYMBOLS];
int      hEMA_H4[MAX_SYMBOLS];
int      hRVI_H4[MAX_SYMBOLS];
int      hATRfast_H4[MAX_SYMBOLS];
int      hATRslow_H4[MAX_SYMBOLS];
datetime gH4LastBar[MAX_SYMBOLS];

double   gRSI_H4v[MAX_SYMBOLS];
double   gMACD_H4v[MAX_SYMBOLS];
double   gEMA_H4v[MAX_SYMBOLS];
double   gRVI_H4v[MAX_SYMBOLS];
double   gATRratio_H4[MAX_SYMBOLS];

int SymbolIndex(const string symbol)
{
   for(int i=0;i<gSymbolCount;i++)
      if(gSymbols[i]==symbol) return i;
   return -1;
}

datetime gLastBasketCloseTime[MAX_SYMBOLS];
datetime gLastBasketCloseExecBar[MAX_SYMBOLS];

void MarkBasketClosedForCooldown(const string symbol)
{
   int symIdx = SymbolIndex(symbol);
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return;
   gLastBasketCloseTime[symIdx] = TimeCurrent();
   datetime t = iTime(symbol, TF_EXEC, 1);
   if(t <= 0) t = TimeCurrent();
   gLastBasketCloseExecBar[symIdx] = t;
}

bool IsReentryCooldownActive(const string symbol,const int symIdx)
{
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return false;
   if(ReentryCooldownSeconds <= 0 && ReentryCooldownExecBars <= 0) return false;

   bool blocked = false;

   if(ReentryCooldownSeconds > 0 && gLastBasketCloseTime[symIdx] > 0)
   {
      if((TimeCurrent() - gLastBasketCloseTime[symIdx]) < ReentryCooldownSeconds)
         blocked = true;
   }

   if(ReentryCooldownExecBars > 0 && gLastBasketCloseExecBar[symIdx] > 0)
   {
      datetime curExecBar = iTime(symbol, TF_EXEC, 1);
      if(curExecBar <= 0) curExecBar = TimeCurrent();
      int barsPassed = iBarShift(symbol, TF_EXEC, gLastBasketCloseExecBar[symIdx], false) - iBarShift(symbol, TF_EXEC, curExecBar, false);
      if(barsPassed < 0) barsPassed = 0;
      if(barsPassed < ReentryCooldownExecBars)
         blocked = true;
   }

   return blocked;
}

void ClearDeepBasketStrongFirstTradeArm(const int symIdx)
{
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return;
   gDeepBasketStrongEntryArmed[symIdx] = false;
   gDeepBasketStrongEntryArmTime[symIdx] = 0;
   gDeepBasketStrongEntryClosedCount[symIdx] = 0;
}

void ArmDeepBasketStrongFirstTrade(const string symbol,const int closedCount)
{
   if(!UseDeepBasketStrongFirstTrade) return;
   if(closedCount < DeepBasketStrongFirstTradeMinClosed) return;

   int symIdx = SymbolIndex(symbol);
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return;

   gDeepBasketStrongEntryArmed[symIdx] = true;
   gDeepBasketStrongEntryArmTime[symIdx] = TimeCurrent();
   gDeepBasketStrongEntryClosedCount[symIdx] = closedCount;

   if(VerboseLogging)
      Print("DEEP BASKET STRONG FIRST TRADE ARMED | symbol=", symbol,
            " closedCount=", closedCount,
            " mult=", DoubleToString(DeepBasketStrongFirstTradeMult, 2));
}

bool ShouldUseDeepBasketStrongFirstTrade(const int symIdx)
{
   if(!UseDeepBasketStrongFirstTrade) return false;
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return false;
   if(!gDeepBasketStrongEntryArmed[symIdx]) return false;
   if(gPositionsCount[symIdx] > 0) return false;
   return true;
}


double Clamp(const double v, const double lo, const double hi)
{
   if(v < lo) return lo;
   if(v > hi) return hi;
   return v;
}

bool Copy1(const int handle, const int buffer, double &outVal)
{
   if(handle == INVALID_HANDLE) return false;
   double tmp[1];
   if(CopyBuffer(handle, buffer, 0, 1, tmp) <= 0) return false;
   outVal = tmp[0];
   return true;
}

void CloneState(const double &src[], double &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}

void BuildPostOpenNextState(const string symbol,
                            const int symIdx,
                            const int positionsAfterOpen,
                            CArrayDouble &trades,
                            double &nextState[])
{
   BuildState(symbol, symIdx, positionsAfterOpen, trades, nextState);

   bool extremeAfter=IsExtremeState(symIdx,positionsAfterOpen);
   if(SeparateExtremeStates)
   {
      int sz=ArraySize(nextState);
      if(sz>0) nextState[sz-1]=(extremeAfter?1.0:0.0);
   }
}

void SubmitTransitionWithNextState(const int symIdx,
                                   const int regime,
                                   const double &state[],
                                   const int action,
                                   const double reward,
                                   const bool done,
                                   const double &nextState[])
{
   double stateCopy[];
   double nextCopy[];
   CloneState(state,stateCopy);
   CloneState(nextState,nextCopy);

   if(UseRealPostActionTransitions)
      ReplayPush(symIdx,regime,state,action,reward,nextState,done);
   else
      ReplayPush(symIdx,regime,state,action,reward,state,done);

   if(!UseReplayBuffer)
   {
      if(UseRealPostActionTransitions)
         DQNUpdate(symIdx,regime,stateCopy,action,reward,nextCopy,done);
      else
         DQNUpdate(symIdx,regime,stateCopy,action,reward,stateCopy,done);
   }
}




void WriteDoubleArray(const int h, const double &arr[])
{
   int sz=ArraySize(arr);
   FileWriteInteger(h, sz);
   for(int i=0;i<sz;i++) FileWriteDouble(h, arr[i]);
}

void ReadDoubleArray(const int h, double &arr[])
{
   int sz=FileReadInteger(h);
   ArrayResize(arr, sz);
   for(int i=0;i<sz;i++) arr[i]=FileReadDouble(h);
}

void WriteBasketSnapshot(const int h, const BasketSnapshot &snap)
{
   FileWriteLong(h, (long)snap.timeStamp);
   FileWriteString(h, snap.symbol);
   FileWriteInteger(h, snap.magic);
   FileWriteInteger(h, snap.regime);
   FileWriteInteger(h, snap.basketDir);
   FileWriteInteger(h, snap.positionsCount);
   FileWriteDouble(h, snap.equity);
   FileWriteDouble(h, snap.openPnL);
   FileWriteDouble(h, snap.avgPrice);
   FileWriteDouble(h, snap.midPrice);
   FileWriteDouble(h, snap.entryPrice);
   FileWriteDouble(h, snap.gridStep);
   FileWriteDouble(h, snap.danger);
   WriteDoubleArray(h, snap.stateKey);
   WriteDoubleArray(h, snap.qVals);
}

void ReadBasketSnapshot(const int h, BasketSnapshot &snap)
{
   snap.timeStamp=(datetime)FileReadLong(h);
   snap.symbol=FileReadString(h);
   snap.magic=FileReadInteger(h);
   snap.regime=FileReadInteger(h);
   snap.basketDir=FileReadInteger(h);
   snap.positionsCount=FileReadInteger(h);
   snap.equity=FileReadDouble(h);
   snap.openPnL=FileReadDouble(h);
   snap.avgPrice=FileReadDouble(h);
   snap.midPrice=FileReadDouble(h);
   snap.entryPrice=FileReadDouble(h);
   snap.gridStep=FileReadDouble(h);
   snap.danger=FileReadDouble(h);
   ReadDoubleArray(h, snap.stateKey);
   ReadDoubleArray(h, snap.qVals);
}

void WriteTickTraceItem(const int h, const TickTraceItem &tick)
{
   FileWriteLong(h, (long)tick.timeStamp);
   FileWriteDouble(h, tick.bid);
   FileWriteDouble(h, tick.ask);
   FileWriteDouble(h, tick.mid);
   FileWriteDouble(h, tick.spreadPoints);
   FileWriteDouble(h, tick.equity);
   FileWriteDouble(h, tick.danger);
}

void ReadTickTraceItem(const int h, TickTraceItem &tick)
{
   tick.timeStamp=(datetime)FileReadLong(h);
   tick.bid=FileReadDouble(h);
   tick.ask=FileReadDouble(h);
   tick.mid=FileReadDouble(h);
   tick.spreadPoints=FileReadDouble(h);
   tick.equity=FileReadDouble(h);
   tick.danger=FileReadDouble(h);
}

double SafeDiv(const double num, const double den, const double fallback=0.0)
{
   if(MathAbs(den) <= 1e-12) return fallback;
   return num / den;
}

void Push(double &arr[], const double v)
{
   int n = ArraySize(arr);
   ArrayResize(arr, n+1);
   arr[n] = v;
}

double NormalizeVolume(const string symbol, double vol)
{
   double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vmax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(vstep <= 0.0) vstep = 0.01;
   if(vmin  <= 0.0) vmin  = vstep;
   if(vmax  <= 0.0) vmax  = vol;

   if(vol < vmin) vol = vmin;
   if(vol > vmax) vol = vmax;

   double steps = MathFloor(vol / vstep + 1e-9);
   double out   = steps * vstep;

   if(out < vmin) out = vmin;
   if(out > vmax) out = vmax;

   int digits = 0;
   if(vstep < 1.0)
   {
      double lg = -MathLog10(vstep);
      if(lg < 0.0) lg = 0.0;
      digits = (int)MathCeil(lg);
      if(digits > 8) digits = 8;
   }
   return NormalizeDouble(out, digits);
}

double Sigmoid(const double x)
{
   double z=MathMax(-60.0, MathMin(60.0, x));
   return 1.0/(1.0+MathExp(-z));
}

void NormalizeVec(double &v[])
{
   double norm=0.0;
   for(int i=0;i<ArraySize(v);i++) norm += v[i]*v[i];
   norm=MathSqrt(norm);
   if(norm<=1e-12) return;
   for(int i=0;i<ArraySize(v);i++) v[i]/=norm;
}
void NormalizeVecN(double &v[]) { NormalizeVec(v); }

void NormalizeVecSlice(double &v[], const int start, const int count)
{
   int n=ArraySize(v);
   int c=MathMax(count,0);
   if(c<=0 || start<0 || start>=n) return;
   int end=MathMin(start+c,n);
   double norm=0.0;
   for(int i=start;i<end;i++) norm += v[i]*v[i];
   norm=MathSqrt(norm);
   if(norm<=1e-12) return;
   for(int i=start;i<end;i++) v[i]/=norm;
}

double CosSim(const double &a[], const double &b[])
{
   int n=ArraySize(a);
   if(n<=0 || ArraySize(b)!=n) return -1.0;
   double dot=0, aa=0, bb=0;
   for(int i=0;i<n;i++){ dot+=a[i]*b[i]; aa+=a[i]*a[i]; bb+=b[i]*b[i]; }
   if(aa<=1e-12 || bb<=1e-12) return -1.0;
   return dot/(MathSqrt(aa)*MathSqrt(bb));
}

bool BuildDeltaVec(const double &cur[], const double &prev[], int n, double &out[])
{
   ArrayResize(out, n);
   double s2=0.0;
   for(int i=0;i<n;i++)
   {
      out[i]=cur[i]-prev[i];
      s2 += out[i]*out[i];
   }
   if(s2<=1e-12) return false;
   NormalizeVec(out);
   return true;
}

double ProtoAgeFactor(const ProtoEntry &p)
{
   if(ProtoHalfLifeDays<=0.0) return 1.0;
   datetime now=TimeCurrent();
   double ageSec=(double)(now - p.created);
   if(ageSec<=0.0) return 1.0;

   double half=ProtoHalfLifeDays*86400.0;
   if(half<=1.0) return 1.0;

   double k = 0.6931471805599453;
   double f = MathExp(-k * (ageSec/half));
   return Clamp(f, 0.05, 1.0);
}

void ProtoScoreBump(int idx, double delta)
{
   if(idx<0 || idx>=ArraySize(gProtos)) return;
   gProtos[idx].survivalScore = MathMax(0.0, gProtos[idx].survivalScore + delta);
}

void PruneProtosIfNeeded()
{
   int n=ArraySize(gProtos);
   if(n<=MaxProtosStored) return;

   while(ArraySize(gProtos) > MaxProtosStored)
   {
      int worst=-1;
      double worstEff=DBL_MAX;
      datetime worstTime=TimeCurrent();

      int m=ArraySize(gProtos);
      for(int i=0;i<m;i++)
      {
         double eff = gProtos[i].survivalScore * ProtoAgeFactor(gProtos[i]);
         datetime t = gProtos[i].created;

         if(gProtos[i].survivalScore >= PruneMinKeepScore)
            eff += 1e6;

         if(eff < worstEff || (MathAbs(eff-worstEff)<1e-9 && t < worstTime))
         {
            worstEff=eff;
            worstTime=t;
            worst=i;
         }
      }

      if(worst<0) worst=0;
      ArrayRemove(gProtos, worst, 1);
   }
}

double QMemAgeFactor(const QMemEntry &e)
{
   if(ProtoHalfLifeDays<=0.0) return 1.0;

   datetime now=TimeCurrent();
   double ageSec=(double)(now - e.created);
   if(ageSec<=0.0) return 1.0;

   double half=ProtoHalfLifeDays*86400.0;
   if(half<=1.0) return 1.0;

   double k=0.6931471805599453;
   double f=MathExp(-k*(ageSec/half));
   return Clamp(f,0.05,1.0);
}

double QMemSimilarity(const double &a[], const double &b[])
{
   return CosSim(a,b);
}

void PruneQMemoryIfNeeded()
{
   if(ArraySize(gQMem)<=MaxQMemEntries) return;

   while(ArraySize(gQMem) > MaxQMemEntries)
   {
      int worst=-1;
      double worstEff=DBL_MAX;
      datetime worstTime=TimeCurrent();

      int n=ArraySize(gQMem);
      for(int i=0;i<n;i++)
      {
         double eff = gQMem[i].score * gQMem[i].conf * QMemAgeFactor(gQMem[i]);
         datetime t = gQMem[i].created;

         if(gQMem[i].score >= QMemPruneMinScore)
            eff += 1e6;

         if(eff < worstEff || (MathAbs(eff-worstEff)<1e-9 && t < worstTime))
         {
            worstEff=eff;
            worstTime=t;
            worst=i;
         }
      }

      if(worst<0) worst=0;
      ArrayRemove(gQMem,worst,1);
   }
}

bool ComputeFingerprint(const string sym, ENUM_TIMEFRAMES tf, int bars, double &out[])
{
   if(bars<20) return false;

   MqlRates rates[];
   int got=CopyRates(sym, tf, 0, bars+2, rates);
   if(got < bars+2) return false;

   double rets[];
   ArrayResize(rets,bars);

   double mean=0.0;
   for(int i=0;i<bars;i++)
   {
      double c0=rates[i].close;
      double c1=rates[i+1].close;
      rets[i]=SafeDiv(c0-c1, c1, 0.0);
      mean += rets[i];
   }
   mean /= bars;

   double var=0.0;
   for(int i=0;i<bars;i++){ double d=rets[i]-mean; var+=d*d; }
   var /= MathMax(1,bars-1);
   double std=MathSqrt(var);

   int fast=10, slow=50;
   double emaF=rates[bars].close, emaS=rates[bars].close;
   double kf=2.0/(fast+1.0), ks=2.0/(slow+1.0);
   for(int i=bars-1;i>=0;i--)
   {
      emaF = rates[i].close*kf + emaF*(1.0-kf);
      emaS = rates[i].close*ks + emaS*(1.0-ks);
   }
   double slope = SafeDiv(emaF-emaS, rates[0].close, 0.0);

   int pos=0, neg=0;
   for(int i=0;i<bars;i++){ if(rets[i]>0) pos++; else if(rets[i]<0) neg++; }
   int dom=(pos>=neg?1:-1);

   int same=0;
   for(int i=0;i<bars;i++){ if(dom==1 && rets[i]>0) same++; if(dom==-1 && rets[i]<0) same++; }
   double monot = SafeDiv(same, bars, 0.0);

   double sameMag=0, oppMag=0;
   for(int i=0;i<bars;i++)
   {
      double r=rets[i];
      if(dom==1){ if(r>0) sameMag+=r; else oppMag+=-r; }
      else      { if(r<0) sameMag+=-r; else oppMag+=r;  }
   }
   double pullback = SafeDiv(oppMag, sameMag+1e-12, 0.0);

   int aFast=14, aSlow=100;
   int nF=MathMin(aFast,bars), nS=MathMin(aSlow,bars);
   double trF=0, trS=0;
   for(int i=0;i<nF;i++) trF += (rates[i].high-rates[i].low);
   for(int i=0;i<nS;i++) trS += (rates[i].high-rates[i].low);
   trF = SafeDiv(trF,nF,0.0);
   trS = SafeDiv(trS,nS,0.0);
   double atrRatio = SafeDiv(trF,trS,1.0);

   double maxAbs=0.0;
   for(int i=0;i<bars;i++) maxAbs=MathMax(maxAbs, MathAbs(rets[i]));
   double impulse=SafeDiv(maxAbs, std+1e-12, 0.0);

   ArrayResize(out,6);
   out[0]=slope;
   out[1]=monot;
   out[2]=pullback;
   out[3]=atrRatio;
   out[4]=impulse;
   out[5]=std;

   NormalizeVec(out);
   return true;
}

// forward
double GetEAEquity();

bool AllowLearnEntryOrAveraging(const int symIdx)
{
   if(!UseDangerBrain) return true;
   if(gMode[symIdx]==MODE_DANGER) return false;
   return true;
}
bool AllowLearnClose(const int symIdx){ return true; }

double ProfitReturnRewardForSymbol(const int symIdx,const double closedProfit)
{
   double denom = GetSymbolVirtualBudgetBase(symIdx);
   if(denom<=1e-9) denom=1000.0;

   double r = closedProfit / denom;
   r = Clamp(r, -ProfitReturnClamp, +ProfitReturnClamp);
   return r * ProfitRewardScale;
}

double RecoveryQualityRewardFromCounts(const int wins, const int losses, const int total)
{
   if(total<=0) return 0.0;
   double winRatio = (double)wins/(double)total;

   double r=0.0;
   r += RecWinRatioScale * winRatio;
   r -= RecLossCountPenalty * (double)losses;
   r -= RecTradesUsedPenalty * (double)total;

   return Clamp(r, -RecQualityClamp, +RecQualityClamp);
}

double CurrentMarginStressRatio()
{
   double budgetBase = (EquityBudget>1e-9 ? EquityBudget : gEAStartEquity);
   if(budgetBase<=1e-9) budgetBase = 1000.0;
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double freeRatio = Clamp(SafeDiv(freeMargin, budgetBase, 0.0), 0.0, 2.0);
   return Clamp(1.0 - freeRatio, 0.0, 1.0);
}

double CurrentReplayRiskBias(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   double danger = Clamp(gDangerReplaySeverityEMA[symIdx], 0.0, 5.0) / 5.0;
   double deep   = Clamp(gDeepBasketReplaySeverityEMA[symIdx], 0.0, 5.0) / 5.0;
   return Clamp(0.6*danger + 0.4*deep, 0.0, 1.0);
}

double BasketAgeDays(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   if(gFirstTradeTime[symIdx] <= 0) return 0.0;
   double ageSec = (double)(TimeCurrent() - gFirstTradeTime[symIdx]);
   if(ageSec <= 0.0) return 0.0;
   return ageSec / 86400.0;
}

double NormalizeMoneyReturnToBudget(const double money,const double clampAbs)
{
   double denom = (EquityBudget>1e-9 ? EquityBudget : gEAStartEquity);
   if(denom<=1e-9) denom=1000.0;

   double lim = MathMax(clampAbs, 1e-6);
   return Clamp(money / denom, -lim, +lim);
}

double GetSymbolVirtualBudgetBase(const int symIdx)
{
   double denom = (EquityBudget>1e-9 ? EquityBudget : gEAStartEquity);
   if(denom<=1e-9) denom=1000.0;

   if(UsePerSymbolVirtualBudget)
   {
      int n=MathMax(1, gSymbolCount);
      denom /= (double)n;
   }

   return MathMax(denom, 100.0);
}

double GetSymbolFloatingDDPct(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   string symbol = gSymbols[symIdx];
   int magic = gMagics[symIdx];
   double ddMoney = GetSymbolFloatingLossMoney(symbol, magic);
   double base = GetSymbolVirtualBudgetBase(symIdx);
   return Clamp(SafeDiv(ddMoney, base, 0.0), 0.0, 1.0);
}

double GetSymbolBasketOpenPnL(const string symbol,const int magic)
{
   double totalPnL=0.0;
   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      totalPnL += PositionGetDouble(POSITION_PROFIT);
   }
   return totalPnL;
}

void ResetDenseBasketHealthTracker(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   gDenseBasketHealth[symIdx].active=false;
   gDenseBasketHealth[symIdx].lastBarTime=0;
   gDenseBasketHealth[symIdx].prevOpenReturn=0.0;
   gDenseBasketHealth[symIdx].prevDD=0.0;
   gDenseBasketHealth[symIdx].prevPositions=0;
   gDenseBasketHealth[symIdx].prevAgeDays=0.0;
}

void PendingAccrueDenseRewardForSymbol(const int symIdx,const double reward)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(MathAbs(reward) <= 1e-12) return;

   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(!gPending[i].active) continue;
      if(gPending[i].symIdx!=symIdx) continue;
      gPending[i].denseRewardAccum += reward;
   }
}

double ComputeDenseBasketHealthReward(const double prevOpenReturn,
                                      const double prevDD,
                                      const int prevPositions,
                                      const double prevAgeDays,
                                      const double curOpenReturn,
                                      const double curDD,
                                      const int curPositions,
                                      const double curAgeDays,
                                      const bool extreme)
{
   double reward=0.0;
   double pnlDelta = curOpenReturn - prevOpenReturn;
   reward += DenseBasketPnLDeltaScale * pnlDelta;

   double ddDelta = Clamp(curDD - prevDD, -1.0, 1.0);
   if(ddDelta > 0.0)
      reward -= DenseBasketDDDeltaPenaltyScale * ddDelta;
   else
      reward += DenseBasketDDRecoveryScale * (-ddDelta);

   int posDelta = MathMax(curPositions - prevPositions, 0);
   if(posDelta > 0)
      reward -= DenseBasketAddPenaltyScale * (double)(posDelta * posDelta);

   double ageDelta = MathMax(curAgeDays - prevAgeDays, 0.0);
   if(ageDelta > 0.0 && pnlDelta <= 0.0)
      reward -= DenseBasketStallAgePenaltyScale * ageDelta;

   if(curPositions <= 1 && pnlDelta > 0.0 && ddDelta <= 0.0)
   {
      double progress = Clamp(pnlDelta / MathMax(DenseBasketOpenReturnClamp, 1e-6), 0.0, 1.0);
      reward += DenseBasketOneRoundProgressScale * progress;
   }

   reward = ScaleRewardByRegime(reward, extreme);
   return Clamp(reward, -DenseBasketRewardClamp, DenseBasketRewardClamp);
}

void AccrueDenseBasketHealthFeedback(const string symbol,
                                     const int symIdx,
                                     const int magic,
                                     const int positionsCount,
                                     const bool extreme)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   if(!UseDenseBasketHealthReward || positionsCount<=0)
   {
      ResetDenseBasketHealthTracker(symIdx);
      return;
   }

   datetime barTime=iTime(symbol, BaseTF, 1);
   if(barTime<=0) return;

   double curOpenPnL = GetSymbolBasketOpenPnL(symbol, magic);
   double curOpenReturn = NormalizeMoneyReturnToBudget(curOpenPnL, DenseBasketOpenReturnClamp);
   double curDD = GetSymbolFloatingDDPct(symIdx);
   double curAgeDays = BasketAgeDays(symIdx);

   if(!gDenseBasketHealth[symIdx].active || gDenseBasketHealth[symIdx].lastBarTime<=0)
   {
      gDenseBasketHealth[symIdx].active=true;
      gDenseBasketHealth[symIdx].lastBarTime=barTime;
      gDenseBasketHealth[symIdx].prevOpenReturn=curOpenReturn;
      gDenseBasketHealth[symIdx].prevDD=curDD;
      gDenseBasketHealth[symIdx].prevPositions=positionsCount;
      gDenseBasketHealth[symIdx].prevAgeDays=curAgeDays;
      return;
   }

   if(gDenseBasketHealth[symIdx].lastBarTime==barTime) return;

   double reward = ComputeDenseBasketHealthReward(gDenseBasketHealth[symIdx].prevOpenReturn,
                                                  gDenseBasketHealth[symIdx].prevDD,
                                                  gDenseBasketHealth[symIdx].prevPositions,
                                                  gDenseBasketHealth[symIdx].prevAgeDays,
                                                  curOpenReturn,
                                                  curDD,
                                                  positionsCount,
                                                  curAgeDays,
                                                  extreme);

   int basketDir = 0;
   if(gActiveBasketEpisodes[symIdx].active)
      basketDir = gActiveBasketEpisodes[symIdx].basketDir;

   double replayDangerRisk=0.0, replayDeepRisk=0.0, replayEfficient=0.0, replayRecentCaution=0.0;
   ComputeReplayRewardProfile(symIdx, basketDir, positionsCount, replayDangerRisk, replayDeepRisk, replayEfficient, replayRecentCaution);

   reward -= RewardV2ReplayDenseAntiPatternScale *
             (0.45 * replayDangerRisk + 0.75 * replayDeepRisk + 0.20 * MathMax(0.0, replayRecentCaution));

   if(positionsCount <= 1 && curOpenReturn >= gDenseBasketHealth[symIdx].prevOpenReturn && curDD <= gDenseBasketHealth[symIdx].prevDD)
      reward += 0.20 * RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 2.0);

   reward = Clamp(reward, -DenseBasketRewardClamp, DenseBasketRewardClamp);

   if(MathAbs(reward) > 1e-10)
   {
      PendingAccrueDenseRewardForSymbol(symIdx, reward);
      if(gActiveBasketEpisodes[symIdx].active)
      {
         gActiveBasketEpisodes[symIdx].rewardAccum += reward;
         gActiveBasketEpisodes[symIdx].maxDD=MathMax(gActiveBasketEpisodes[symIdx].maxDD, curDD);
         gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
      }
   }

   gDenseBasketHealth[symIdx].active=true;
   gDenseBasketHealth[symIdx].lastBarTime=barTime;
   gDenseBasketHealth[symIdx].prevOpenReturn=curOpenReturn;
   gDenseBasketHealth[symIdx].prevDD=curDD;
   gDenseBasketHealth[symIdx].prevPositions=positionsCount;
   gDenseBasketHealth[symIdx].prevAgeDays=curAgeDays;
}

double SymbolPipSize(const string symbol)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   int digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   if(digits==3 || digits==5) return point*10.0;
   return point;
}

double PipsToPriceDistance(const string symbol,const double pips)
{
   return MathMax(pips,0.0) * SymbolPipSize(symbol);
}

bool GetLatestBasketEntryInfo(const string symbol,
                              const int magic,
                              const int basketDir,
                              double &entryPrice,
                              datetime &entryTime)
{
   entryPrice=0.0;
   entryTime=0;
   if(basketDir==0) return false;

   int desiredType=(basketDir>0 ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);
   bool found=false;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      if((int)PositionGetInteger(POSITION_TYPE)!=desiredType) continue;

      datetime t=(datetime)PositionGetInteger(POSITION_TIME);
      double px=PositionGetDouble(POSITION_PRICE_OPEN);

      if(!found || t>entryTime)
      {
         found=true;
         entryTime=t;
         entryPrice=px;
      }
   }

   return found;
}

double ComputeEffectiveMinAddSpacing(const string symbol,const double refPrice)
{
   double fixedDist = PipsToPriceDistance(symbol, MinAddSpacingPips);
   double atr = GetATRValueTF(symbol, BaseTF, 14, 1);
   double atrDist = MathMax(MinAddSpacingATRFrac, 0.0) * MathMax(atr, 0.0);
   double pctDist = MathMax(MinAddSpacingPricePct, 0.0) * MathAbs(refPrice);
   return MathMax(fixedDist, MathMax(atrDist, pctDist));
}

bool PassesMinAddSpacingGate(const string symbol,
                             const int symIdx,
                             const int magic,
                             const int basketDir,
                             const double candidatePrice,
                             double &requiredGap,
                             double &actualGap,
                             double &lastEntryPrice)
{
   requiredGap=0.0;
   actualGap=0.0;
   lastEntryPrice=0.0;

   if(!UseMinAddSpacingGate) return true;
   if(basketDir==0) return true;
   if(candidatePrice<=0.0) return true;

   datetime lastEntryTime=0;
   if(!GetLatestBasketEntryInfo(symbol, magic, basketDir, lastEntryPrice, lastEntryTime))
      return true;

   double refPrice = (lastEntryPrice>0.0 ? lastEntryPrice : candidatePrice);
   double minSpacing = ComputeEffectiveMinAddSpacing(symbol, refPrice);
   if(minSpacing<0.0) minSpacing=0.0;

   double baseStep=0.0;
   double adaptiveStep=baseStep;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
   {
      baseStep=gGridStepCache[symIdx];
      adaptiveStep=gGridActiveStep[symIdx];
   }

   if(baseStep<0.0) baseStep=0.0;
   if(adaptiveStep<baseStep) adaptiveStep=baseStep;

   double extraAdaptiveGap = adaptiveStep - baseStep;
   if(extraAdaptiveGap<0.0) extraAdaptiveGap=0.0;

   requiredGap = minSpacing + extraAdaptiveGap;
   if(requiredGap<=0.0) return true;

   actualGap = MathAbs(candidatePrice - lastEntryPrice);
   return (actualGap + 1e-12 >= requiredGap);
}

bool PassesOpenIntervalGate(const string symbol,
                           const int symIdx)
{
   if(!UseMinOpenIntervalGate) return true;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return true;

   datetime now = TimeCurrent();
   datetime lastOpen = gLastTradeOpenTime[symIdx];
   if(lastOpen>0)
   {
      int minSec = MathMax(0, MinSecondsBetweenOpens);
      if(minSec>0 && (now - lastOpen) < minSec)
         return false;
   }

   int minBars = MathMax(0, MinExecBarsBetweenOpens);
   if(minBars>0)
   {
      datetime lastBar = iTime(symbol, TF_EXEC, minBars);
      if(lastBar>0 && lastOpen>=lastBar)
         return false;
   }

   return true;
}



double ArchiveEpisodeBaseWeight(const EpisodeMemory &e,
                                const string symbol,
                                const int regimeType,
                                const int patternType,
                                const int liquidityType,
                                const int basketDir,
                                const int positionsCount,
                                const int idx,
                                const int startIdx,
                                const int endIdx)
{
   if(e.symbol != symbol) return 0.0;

   double w = 0.20;
   if(e.regimeType == regimeType) w += 0.25;
   else if(MathAbs(e.regimeType - regimeType) == 1) w += 0.12;

   if(e.patternType == patternType)   w += 0.20;
   if(e.liquidityType == liquidityType) w += 0.15;
   if(basketDir != 0 && e.basketDir == basketDir) w += 0.10;

   int posGap = MathAbs(e.openPositionsMax - MathMax(1, positionsCount));
   w *= SafeDiv(1.0, 1.0 + 0.35 * (double)posGap, 1.0);

   double progress = SafeDiv((double)(idx - startIdx + 1), (double)MathMax(1, endIdx - startIdx + 1), 1.0);
   w *= (0.65 + 0.35 * Clamp(progress, 0.0, 1.0));
   return w;
}

void ComputeArchiveEpisodeProfile(const int symIdx,
                                  const int basketDir,
                                  const int positionsCount,
                                  double &badRiskOut,
                                  double &cleanQualityOut,
                                  double &deepRiskOut,
                                  double &efficiencyOut)
{
   badRiskOut=0.0;
   cleanQualityOut=0.0;
   deepRiskOut=0.0;
   efficiencyOut=0.0;

   if(!UseArchiveAwareReward) return;

   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, patternType, liquidityType, sessionType, atrRatio);

   int scan = MathMax(50, ArchiveRewardScanLimit);
   int startIdx = MathMax(0, n - scan);

   double sumW=0.0;
   double badAcc=0.0, cleanAcc=0.0, deepAcc=0.0, effAcc=0.0;

   for(int i=startIdx;i<n;i++)
   {
      double w = ArchiveEpisodeBaseWeight(gEpisodeMemory[i], symbol, regimeType, patternType, liquidityType,
                                          basketDir, positionsCount, i, startIdx, n-1);
      if(w <= 0.10) continue;

      double bad = 0.0;
      bad += Clamp(gEpisodeMemory[i].maxDrawdownPct / MathMax(RewardV2CleanCycleMaxDD * 2.0, 0.01), 0.0, 4.0);
      bad += 0.30 * (double)MathMax(0, gEpisodeMemory[i].addCount);
      bad += 0.20 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);
      if(gEpisodeMemory[i].rewardEfficiency < 0.0)
         bad += Clamp(-gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);
      if(gEpisodeMemory[i].inefficientRecovery != 0) bad += 0.75;
      if(gEpisodeMemory[i].forcedStopLikeEvent != 0) bad += 0.75;

      double deep = Clamp(gEpisodeMemory[i].maxDrawdownPct / 0.05, 0.0, 5.0);
      deep += 0.20 * (double)MathMax(0, gEpisodeMemory[i].addCount);
      deep += 0.15 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);

      double clean = 0.0;
      if(gEpisodeMemory[i].oneRoundTrade != 0 && gEpisodeMemory[i].rewardTotal > 0.0)
         clean = Clamp(1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct, MathMax(RewardV2CleanCycleMaxDD, 1e-6), 1.0), 0.0, 1.0);

      double eff = Clamp(gEpisodeMemory[i].rewardEfficiency, -2.0, 2.0);

      badAcc += w * bad;
      cleanAcc += w * clean;
      deepAcc += w * deep;
      effAcc += w * eff;
      sumW += w;
   }

   if(sumW <= 1e-9) return;

   badRiskOut = Clamp(badAcc / sumW, 0.0, 4.0);
   cleanQualityOut = Clamp(cleanAcc / sumW, 0.0, 1.0);
   deepRiskOut = Clamp(deepAcc / sumW, 0.0, 5.0);
   efficiencyOut = Clamp(effAcc / sumW, -2.0, 2.0);
}

void ComputeArchivePatternAndRegimePenalty(const int symIdx,
                                           double &patternPenaltyOut,
                                           double &regimePenaltyOut)
{
   patternPenaltyOut=0.0;
   regimePenaltyOut=0.0;

   if(!UseArchiveAwareReward) return;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, patternType, liquidityType, sessionType, atrRatio);

   int scan = MathMax(50, ArchiveRewardScanLimit);

   int np=ArraySize(gPatternMemory);
   if(np>0)
   {
      int sp=MathMax(0, np-scan);
      double sumW=0.0, penAcc=0.0;
      for(int i=sp;i<np;i++)
      {
         if(gPatternMemory[i].symbol != symbol) continue;

         double w=0.20;
         if(gPatternMemory[i].patternType == patternType) w += 0.35;
         if(gPatternMemory[i].volatilityClass == regimeType) w += 0.25;
         if(gPatternMemory[i].liquidityClass == liquidityType) w += 0.20;
         double progress = SafeDiv((double)(i-sp+1), (double)MathMax(1, np-sp), 1.0);
         w *= (0.65 + 0.35 * Clamp(progress, 0.0, 1.0));

         double pen = Clamp(gPatternMemory[i].ddMean / 0.05, 0.0, 4.0);
         pen += 0.20 * Clamp(gPatternMemory[i].addMean, 0.0, 6.0);
         if(gPatternMemory[i].strengthClass == 0) pen += 0.50;
         if(gPatternMemory[i].rewardEfficiencyMean < 0.0)
            pen += Clamp(-gPatternMemory[i].rewardEfficiencyMean, 0.0, 2.0);

         penAcc += w * pen;
         sumW += w;
      }
      if(sumW > 1e-9)
         patternPenaltyOut = Clamp(penAcc / sumW, 0.0, 4.0);
   }

   int nr=ArraySize(gRegimeEventMemory);
   if(nr>0)
   {
      int sr=MathMax(0, nr-scan);
      double sumW=0.0, penAcc=0.0;
      for(int i=sr;i<nr;i++)
      {
         if(gRegimeEventMemory[i].symbol != symbol) continue;

         double w=0.10;
         if(gRegimeEventMemory[i].regimeType == regimeType) w += 0.45;
         else if(MathAbs(gRegimeEventMemory[i].regimeType - regimeType) == 1) w += 0.20;
         double progress = SafeDiv((double)(i-sr+1), (double)MathMax(1, nr-sr), 1.0);
         w *= (0.65 + 0.35 * Clamp(progress, 0.0, 1.0));

         double pen = 0.0;
         if(gRegimeEventMemory[i].eventType != 0) pen += 1.00;
         pen += Clamp(gRegimeEventMemory[i].avgSpread / 25.0, 0.0, 1.5);
         if(gRegimeEventMemory[i].avgRewardEfficiency < 0.0)
            pen += Clamp(-gRegimeEventMemory[i].avgRewardEfficiency, 0.0, 2.0) * 0.75;

         penAcc += w * pen;
         sumW += w;
      }
      if(sumW > 1e-9)
         regimePenaltyOut = Clamp(penAcc / sumW, 0.0, 3.0);
   }

}

double ArchiveLiveRecencyWeight(const datetime memEndTime,
                                const int idx,
                                const int startIdx,
                                const int endIdx)
{
   double halfLife = MathMax(ArchiveLiveRecentHalfLifeDays, 0.5);
   double ageDays = MathMax(0.0, (double)(TimeCurrent() - memEndTime) / 86400.0);
   double decay = MathExp(-0.6931471805599453 * ageDays / halfLife);

   double progress = SafeDiv((double)(idx - startIdx + 1),
                             (double)MathMax(1, endIdx - startIdx + 1),
                             1.0);
   double progressW = 0.55 + 0.45 * Clamp(progress, 0.0, 1.0);
   return Clamp(decay * progressW, 0.05, 1.50);
}

double ArchiveLiveDecisionConfidence(const double &q[])
{
   int n=ArraySize(q);
   if(n<=1) return 0.0;

   int best=0;
   int second=1;
   if(q[second] > q[best]){ int t=best; best=second; second=t; }

   for(int i=2;i<n;i++)
   {
      if(q[i] > q[best])
      {
         second=best;
         best=i;
      }
      else if(i!=best && q[i] > q[second])
      {
         second=i;
      }
   }

   double gap = q[best] - q[second];
   double scale=1.0;
   for(int i=0;i<n;i++) scale += MathAbs(q[i]);
   scale /= (double)n;

   return Clamp(gap / MathMax(scale, 1e-6), 0.0, 1.0);
}


bool BuildArchiveLiveActionBias(const int symIdx,
                                const int regime,
                                const double &state[],
                                const double &qIn[],
                                double &biasOut[],
                                double &blendAlphaOut)
{
   ArrayResize(biasOut, ActionCount);
   for(int a=0;a<ActionCount;a++) biasOut[a]=0.0;
   blendAlphaOut=0.0;

   if(!UseArchiveLiveSimilarity) return false;
   if(symIdx<0 || symIdx>=gSymbolCount) return false;
   if(ActionCount < 3) return false;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, regime, regimeType, patternType, liquidityType, sessionType, atrRatio);

   int scanBase = MathMax(80, ArchiveLiveScanLimit);

   double dirScore[3];
   double dirWeight[3];
   for(int k=0;k<3;k++)
   {
      dirScore[k]=0.0;
      dirWeight[k]=0.0;
   }

   double holdPenaltyAcc=0.0;
   double holdBonusAcc=0.0;
   double holdWeight=0.0;

   int ne=ArraySize(gEpisodeMemory);
   if(ne>0)
   {
      int se=MathMax(0, ne-scanBase);
      for(int i=se;i<ne;i++)
      {
         if(gEpisodeMemory[i].symbol != symbol) continue;

         double recW = ArchiveLiveRecencyWeight(gEpisodeMemory[i].endTime, i, se, ne-1);

         double ctxW = 0.20;
         if(gEpisodeMemory[i].regimeType == regimeType) ctxW += 0.24;
         else if(MathAbs(gEpisodeMemory[i].regimeType - regimeType) == 1) ctxW += 0.10;
         if(gEpisodeMemory[i].patternType == patternType) ctxW += 0.22;
         if(gEpisodeMemory[i].liquidityType == liquidityType) ctxW += 0.16;
         if(gEpisodeMemory[i].sessionType == sessionType) ctxW += 0.10;

         double risk = 0.0;
         risk += Clamp(gEpisodeMemory[i].maxDrawdownPct / MathMax(RewardV2CleanCycleMaxDD * 2.0, 0.01), 0.0, 4.0);
         risk += 0.28 * (double)MathMax(0, gEpisodeMemory[i].addCount);
         risk += 0.18 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);
         if(gEpisodeMemory[i].inefficientRecovery != 0) risk += 0.65;
         if(gEpisodeMemory[i].forcedStopLikeEvent != 0) risk += 0.75;
         if(gEpisodeMemory[i].rewardEfficiency < 0.0)
            risk += Clamp(-gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);

         double clean = 0.0;
         if(gEpisodeMemory[i].rewardTotal > 0.0)
         {
            clean += 0.35 * Clamp(gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);
            if(gEpisodeMemory[i].oneRoundTrade != 0)
               clean += Clamp(1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct,
                                            MathMax(RewardV2CleanCycleMaxDD, 1e-6), 1.0), 0.0, 1.0);
         }

         double signedQuality = 0.45 * clean - 0.65 * risk;
         if(gEpisodeMemory[i].rewardEfficiency > 0.0)
            signedQuality += 0.18 * Clamp(gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);

         int actionDir = (gEpisodeMemory[i].basketDir>0 ? 1 : (gEpisodeMemory[i].basketDir<0 ? 2 : 0));
         double w = ctxW * recW;

         if(actionDir>=1 && actionDir<=2)
         {
            dirScore[actionDir]  += w * signedQuality;
            dirWeight[actionDir] += w;
         }

         holdPenaltyAcc += w * MathMax(0.0, risk - 0.35 * clean);
         holdBonusAcc   += w * MathMax(0.0, clean - 0.25 * risk);
         holdWeight     += w;
      }
   }

   double patternCaution=0.0;
   double patternSupport=0.0;
   int np=ArraySize(gPatternMemory);
   if(np>0)
   {
      int sp=MathMax(0, np-scanBase);
      double sumW=0.0;
      for(int i=sp;i<np;i++)
      {
         if(gPatternMemory[i].symbol != symbol) continue;

         double recW = ArchiveLiveRecencyWeight(gPatternMemory[i].endTime, i, sp, np-1);
         double w=0.18;
         if(gPatternMemory[i].patternType == patternType) w += 0.35;
         if(gPatternMemory[i].volatilityClass == regimeType) w += 0.24;
         if(gPatternMemory[i].liquidityClass == liquidityType) w += 0.18;
         w *= recW;

         double caution = Clamp(gPatternMemory[i].ddMean / 0.05, 0.0, 4.0);
         caution += 0.18 * Clamp(gPatternMemory[i].addMean, 0.0, 6.0);
         if(gPatternMemory[i].strengthClass == 0) caution += 0.45;
         if(gPatternMemory[i].rewardEfficiencyMean < 0.0)
            caution += Clamp(-gPatternMemory[i].rewardEfficiencyMean, 0.0, 2.0);

         double support = 0.0;
         if(gPatternMemory[i].strengthClass == 2)
            support += 0.45;
         if(gPatternMemory[i].rewardEfficiencyMean > 0.0)
            support += 0.35 * Clamp(gPatternMemory[i].rewardEfficiencyMean, 0.0, 2.0);
         support -= 0.15 * Clamp(gPatternMemory[i].addMean, 0.0, 6.0);

         patternCaution += w * MathMax(0.0, caution);
         patternSupport += w * MathMax(0.0, support);
         sumW += w;
      }

      if(sumW > 1e-9)
      {
         patternCaution = Clamp(patternCaution / sumW, 0.0, 4.0);
         patternSupport = Clamp(patternSupport / sumW, 0.0, 2.0);
      }
   }

   double regimeCaution=0.0;
   int nr=ArraySize(gRegimeEventMemory);
   if(nr>0)
   {
      int sr=MathMax(0, nr-scanBase);
      double sumW=0.0;
      for(int i=sr;i<nr;i++)
      {
         if(gRegimeEventMemory[i].symbol != symbol) continue;

         double recW = ArchiveLiveRecencyWeight(gRegimeEventMemory[i].endTime, i, sr, nr-1);
         double w=0.10;
         if(gRegimeEventMemory[i].regimeType == regimeType) w += 0.45;
         else if(MathAbs(gRegimeEventMemory[i].regimeType - regimeType) == 1) w += 0.18;
         w *= recW;

         double caution = 0.0;
         if(gRegimeEventMemory[i].eventType != 0) caution += 1.00;
         caution += Clamp(gRegimeEventMemory[i].avgSpread / 25.0, 0.0, 1.5);
         caution += 0.20 * Clamp(MathAbs(gRegimeEventMemory[i].avgVol - GetAtrRatioCached_Base(symbol)), 0.0, 1.5);
         if(gRegimeEventMemory[i].avgRewardEfficiency < 0.0)
            caution += 0.75 * Clamp(-gRegimeEventMemory[i].avgRewardEfficiency, 0.0, 2.0);

         regimeCaution += w * caution;
         sumW += w;
      }

      if(sumW > 1e-9)
         regimeCaution = Clamp(regimeCaution / sumW, 0.0, 3.0);
   }

   double efficientSupport=0.0;
   double efficientCaution=0.0;
   if(UseEfficientPeriodLiveSupport)
   {
      int neff=ArraySize(gEfficientReplayBank.periods);
      if(neff>0)
      {
         int seff=MathMax(0, neff-scanBase);
         double sumW=0.0;
         for(int i=seff;i<neff;i++)
         {
            if(gEfficientReplayBank.periods[i].symIdx != symIdx) continue;

            double recW = ArchiveLiveRecencyWeight(gEfficientReplayBank.periods[i].endTime, i, seff, neff-1);
            double w=0.16;
            if(gEfficientReplayBank.periods[i].regimeType == regimeType) w += 0.28;
            else if(MathAbs(gEfficientReplayBank.periods[i].regimeType - regimeType) == 1) w += 0.12;
            if(gEfficientReplayBank.periods[i].patternType == patternType) w += 0.20;
            if(gEfficientReplayBank.periods[i].liquidityType == liquidityType) w += 0.14;
            if(gEfficientReplayBank.periods[i].sessionType == sessionType) w += 0.10;
            w *= recW;

            double support = 0.0;
            support += Clamp(gEfficientReplayBank.periods[i].rewardEfficiency, -2.0, 2.0);
            support += 0.20 * Clamp((double)gEfficientReplayBank.periods[i].oneRoundCount, 0.0, 6.0);
            support -= 0.18 * Clamp(gEfficientReplayBank.periods[i].ddMax / 0.05, 0.0, 4.0);
            support -= 0.08 * Clamp((double)gEfficientReplayBank.periods[i].addCountTotal, 0.0, 10.0);

            efficientSupport += w * MathMax(0.0, support);
            efficientCaution += w * MathMax(0.0, -support);
            sumW += w;
         }
         if(sumW > 1e-9)
         {
            efficientSupport = Clamp(efficientSupport / sumW, 0.0, 2.5);
            efficientCaution = Clamp(efficientCaution / sumW, 0.0, 2.5);
         }
      }
   }

   double holdBias = 0.0;
   if(holdWeight > 1e-9)
      holdBias = Clamp((holdPenaltyAcc - 0.60 * holdBonusAcc) / holdWeight, -2.0, 3.0);

   holdBias += EfficientPeriodLiveSupportScale * (0.60 * efficientCaution - 0.35 * efficientSupport);

   for(int a=1;a<=2;a++)
   {
      if(dirWeight[a] > 1e-9)
         biasOut[a] += Clamp(dirScore[a] / dirWeight[a], -2.5, 2.5);
   }

   double patternAdj = ArchiveLivePatternCautionScale * (patternCaution - 0.45 * patternSupport);
   double regimeAdj = ArchiveLiveRegimeCautionScale * regimeCaution;

   biasOut[0] += holdBias + patternAdj + regimeAdj;
   double efficientTradeAdj = EfficientPeriodLiveSupportScale * (0.45 * efficientSupport - 0.35 * efficientCaution);
   biasOut[1] += efficientTradeAdj - 0.50 * (patternAdj + regimeAdj);
   biasOut[2] += efficientTradeAdj - 0.50 * (patternAdj + regimeAdj);

   double mean=0.0;
   for(int a=0;a<ActionCount;a++) mean += biasOut[a];
   mean /= (double)ActionCount;
   double maxAbs=0.0;
   for(int a=0;a<ActionCount;a++)
   {
      biasOut[a] -= mean;
      maxAbs = MathMax(maxAbs, MathAbs(biasOut[a]));
   }
   if(maxAbs <= 1e-9) return false;

   for(int a=0;a<ActionCount;a++)
      biasOut[a] = 1.5 * biasOut[a] / maxAbs;

   double conf = ArchiveLiveDecisionConfidence(qIn);
   double uncertainty = Clamp(1.0 - conf, 0.0, 1.0);
   blendAlphaOut = Clamp(ArchiveLiveBlendWeight * (0.35 + 0.65 * uncertainty),
                         ArchiveLiveMinBlendWeight,
                         ArchiveLiveBlendWeight);

   return true;
}

double ComputeArchiveAddRiskScore(const int symIdx,
                                  const int basketDir,
                                  const int positionsCount)
{
   if(!UseArchiveAddRiskGate) return 0.0;
   if(symIdx<0 || symIdx>=gSymbolCount) return 0.0;
   if(basketDir==0) return 0.0;
   if(positionsCount < ArchiveAddRiskGateMinPositions) return 0.0;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, patternType, liquidityType, sessionType, atrRatio);
   int candidatePositions = positionsCount + 1;
   int scanBase = MathMax(80, ArchiveLiveScanLimit);

   double sumW=0.0;
   double riskAcc=0.0;
   double cleanAcc=0.0;

   int ne=ArraySize(gEpisodeMemory);
   if(ne>0)
   {
      int se=MathMax(0, ne-scanBase);
      for(int i=se;i<ne;i++)
      {
         if(gEpisodeMemory[i].symbol != symbol) continue;
         if(gEpisodeMemory[i].basketDir != basketDir) continue;

         double w=0.22;
         if(gEpisodeMemory[i].regimeType == regimeType) w += 0.24;
         if(gEpisodeMemory[i].patternType == patternType) w += 0.22;
         if(gEpisodeMemory[i].liquidityType == liquidityType) w += 0.14;
         if(gEpisodeMemory[i].sessionType == sessionType) w += 0.10;

         int posGap=MathAbs(gEpisodeMemory[i].openPositionsMax - candidatePositions);
         w *= SafeDiv(1.0, 1.0 + 0.30 * (double)posGap, 1.0);
         w *= ArchiveLiveRecencyWeight(gEpisodeMemory[i].endTime, i, se, ne-1);
         if(w<=0.05) continue;

         double risk = 0.0;
         risk += Clamp(gEpisodeMemory[i].maxDrawdownPct / 0.05, 0.0, 5.0);
         risk += 0.26 * (double)MathMax(0, gEpisodeMemory[i].addCount);
         risk += 0.18 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);
         if(gEpisodeMemory[i].inefficientRecovery != 0) risk += 0.75;
         if(gEpisodeMemory[i].forcedStopLikeEvent != 0) risk += 0.70;
         if(gEpisodeMemory[i].rewardEfficiency < 0.0)
            risk += Clamp(-gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);

         double clean = 0.0;
         if(gEpisodeMemory[i].rewardTotal > 0.0)
         {
            clean += 0.30 * Clamp(gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);
            if(gEpisodeMemory[i].oneRoundTrade != 0) clean += 0.60;
         }

         riskAcc += w * risk;
         cleanAcc += w * clean;
         sumW += w;
      }
   }

   double patternPenalty=0.0, regimePenalty=0.0;
   ComputeArchivePatternAndRegimePenalty(symIdx, patternPenalty, regimePenalty);

   double deepSeqRisk=0.0;
   int nds=ArraySize(gDeepBasketReplayBank.sequences);
   if(nds>0)
   {
      int sd=MathMax(0, nds-scanBase);
      double sumWSeq=0.0;
      for(int i=sd;i<nds;i++)
      {
         if(gDeepBasketReplayBank.sequences[i].symIdx != symIdx) continue;
         if(gDeepBasketReplayBank.sequences[i].basketDir != basketDir) continue;

         int posGap=MathAbs(gDeepBasketReplayBank.sequences[i].maxPositions - candidatePositions);
         double w=(0.25 + 0.15 / (1.0 + (double)posGap));
         w *= ArchiveLiveRecencyWeight(gDeepBasketReplayBank.sequences[i].endTime, i, sd, nds-1);

         double seqRisk = Clamp(gDeepBasketReplayBank.sequences[i].maxDD / 0.05, 0.0, 5.0);
         seqRisk += 0.25 * Clamp((double)gDeepBasketReplayBank.sequences[i].addCount, 0.0, 8.0);
         if(gDeepBasketReplayBank.sequences[i].finalReward < 0.0)
            seqRisk += Clamp(-gDeepBasketReplayBank.sequences[i].finalReward, 0.0, 2.0);

         deepSeqRisk += w * seqRisk;
         sumWSeq += w;
      }
      if(sumWSeq > 1e-9)
         deepSeqRisk = Clamp(deepSeqRisk / sumWSeq, 0.0, 5.0);
   }

   double efficientOffset=0.0;
   if(UseEfficientPeriodLiveSupport)
   {
      int neff=ArraySize(gEfficientReplayBank.periods);
      if(neff>0)
      {
         int seff=MathMax(0, neff-scanBase);
         double sumWEff=0.0;
         for(int i=seff;i<neff;i++)
         {
            if(gEfficientReplayBank.periods[i].symIdx != symIdx) continue;
            double w=0.16;
            if(gEfficientReplayBank.periods[i].regimeType == regimeType) w += 0.24;
            if(gEfficientReplayBank.periods[i].patternType == patternType) w += 0.18;
            if(gEfficientReplayBank.periods[i].liquidityType == liquidityType) w += 0.12;
            if(gEfficientReplayBank.periods[i].sessionType == sessionType) w += 0.10;
            w *= ArchiveLiveRecencyWeight(gEfficientReplayBank.periods[i].endTime, i, seff, neff-1);

            double eff = Clamp(gEfficientReplayBank.periods[i].rewardEfficiency, -2.0, 2.0);
            eff -= 0.20 * Clamp(gEfficientReplayBank.periods[i].ddMax / 0.05, 0.0, 4.0);
            eff -= 0.08 * Clamp((double)gEfficientReplayBank.periods[i].addCountTotal, 0.0, 8.0);
            efficientOffset += w * eff;
            sumWEff += w;
         }
         if(sumWEff > 1e-9)
            efficientOffset = Clamp(efficientOffset / sumWEff, -2.0, 2.0);
      }
   }

   double histRisk = 0.0;
   if(sumW > 1e-9)
      histRisk = MathMax(0.0, (riskAcc / sumW) - ArchiveAddRiskGateCleanOffset * (cleanAcc / sumW));

   double currentRisk = 0.0;
   currentRisk += 1.50 * GetSymbolFloatingDDPct(symIdx);
   currentRisk += 0.75 * CurrentReplayRiskBias(symIdx);
   currentRisk += 0.20 * (double)MathMax(0, positionsCount - 1);

   return histRisk + DeepSequenceAddRiskScale * deepSeqRisk + 0.30 * patternPenalty + 0.22 * regimePenalty + currentRisk - 0.18 * MathMax(0.0, efficientOffset);
}
double RewardV2RiskyProfitPenalty(const double profitRewardNorm,
                                  const int addCount,
                                  const int maxPositions,
                                  const double episodeMaxDD,
                                  const double archiveDeepRisk,
                                  const double archiveBadRisk)
{
   if(profitRewardNorm <= 0.0) return 0.0;

   double risk = 0.0;
   risk += 1.40 * Clamp(episodeMaxDD, 0.0, 1.0);
   risk += 0.18 * (double)MathMax(0, addCount);
   risk += 0.14 * (double)MathMax(0, maxPositions - 1);
   risk += 0.10 * Clamp(archiveDeepRisk, 0.0, 5.0);
   risk += 0.08 * Clamp(archiveBadRisk, 0.0, 4.0);
   return profitRewardNorm * risk;
}

void GetRewardEpisodeContext(const int symIdx,
                             const int positionsFallback,
                             int &episodeAdds,
                             int &episodeMaxPositions,
                             double &episodeMaxDD)
{
   episodeAdds = MathMax(positionsFallback - 1, 0);
   episodeMaxPositions = MathMax(positionsFallback, 1);
   episodeMaxDD = GetSymbolFloatingDDPct(symIdx);

   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gActiveBasketEpisodes[symIdx].active)
   {
      episodeAdds = MathMax(episodeAdds, gActiveBasketEpisodes[symIdx].addCount);
      episodeMaxPositions = MathMax(episodeMaxPositions, gActiveBasketEpisodes[symIdx].maxPositions);
      episodeMaxDD = MathMax(episodeMaxDD, gActiveBasketEpisodes[symIdx].maxDD);
   }
}

double RewardV2QuadraticAddPenalty(const int addCount)
{
   if(addCount <= 0) return 0.0;
   return RewardV2AddPenaltyScale * (double)addCount
        + RewardV2AddPenaltyQuadratic * (double)(addCount * addCount);
}

double RewardV2DeepBasketPenalty(const int positionsCount)
{
   int excess = MathMax(positionsCount - RewardV2DeepBasketStartPositions, 0);
   if(excess <= 0) return 0.0;
   return RewardV2DeepBasketPenaltyScale * (double)(excess * excess);
}

double RewardV2ProfitToDDQualityForSymbol(const int symIdx, const double closedProfit, const double episodeMaxDD)
{
   if(closedProfit <= 0.0) return 0.0;
   double base = MathAbs(ProfitReturnRewardForSymbol(symIdx, closedProfit));
   double ddPenaltyDenom = 1.0 + 12.0 * Clamp(episodeMaxDD, 0.0, 1.0);
   return RewardV2ProfitToMaxDDScale * Clamp(base / ddPenaltyDenom, 0.0, RewardV2RewardClamp);
}



double ComputeCloseRewardV2(const int symIdx,
                            const BasketCloseResult &closeRes,
                            const int positionsBeforeClose,
                            const bool extreme)
{
   double reward = 0.0;

   int episodeAdds = 0;
   int episodeMaxPositions = 1;
   double episodeMaxDD = 0.0;
   GetRewardEpisodeContext(symIdx, positionsBeforeClose, episodeAdds, episodeMaxPositions, episodeMaxDD);

   int basketDir = 0;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gActiveBasketEpisodes[symIdx].active)
      basketDir = gActiveBasketEpisodes[symIdx].basketDir;

   double currentDD = GetSymbolFloatingDDPct(symIdx);
   double replayRisk = CurrentReplayRiskBias(symIdx);
   double marginStress = CurrentMarginStressRatio();
   double ageDays = BasketAgeDays(symIdx);

   double archiveBadRisk=0.0, archiveCleanQuality=0.0, archiveDeepRisk=0.0, archiveEfficiency=0.0;
   ComputeArchiveEpisodeProfile(symIdx, basketDir, episodeMaxPositions, archiveBadRisk, archiveCleanQuality, archiveDeepRisk, archiveEfficiency);

   double archivePatternPenalty=0.0, archiveRegimePenalty=0.0;
   ComputeArchivePatternAndRegimePenalty(symIdx, archivePatternPenalty, archiveRegimePenalty);

   double replayDangerRisk=0.0, replayDeepRisk=0.0, replayEfficient=0.0, replayRecentCaution=0.0;
   ComputeReplayRewardProfile(symIdx, basketDir, episodeMaxPositions, replayDangerRisk, replayDeepRisk, replayEfficient, replayRecentCaution);

   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double recentCalm=ComputeRecentDrawdownCalmBonus(symIdx);
   double periodicQualityTilt=Clamp((recentQuality - 0.5)*2.0,-1.0,1.0);

   double profitRewardNorm = ProfitReturnRewardForSymbol(symIdx, closeRes.closedProfit);

   reward += profitRewardNorm * RewardV2BasketProfitScale;
   reward += RewardV2RecoveryQualityScale * RecoveryQualityRewardFromCounts(closeRes.wins, closeRes.losses, closeRes.total);
   reward += RewardV2ProfitToDDQualityForSymbol(symIdx, closeRes.closedProfit, episodeMaxDD);
   reward += 0.22 * periodicQualityTilt;

   if(closeRes.closedProfit > 0.0 && episodeMaxPositions <= 1)
      reward += RewardV2OneRoundBonus;

   if(closeRes.closedProfit > 0.0 &&
      episodeMaxPositions <= 1 &&
      episodeMaxDD <= RewardV2CleanCycleMaxDD)
   {
      reward += RewardV2CleanCycleBonus;
   }

   reward -= RewardV2QuadraticAddPenalty(episodeAdds);
   reward -= RewardV2DeepBasketPenalty(episodeMaxPositions);
   reward -= RewardV2EpisodeMaxDDPenaltyScale * Clamp(episodeMaxDD, 0.0, 1.0);
   reward -= RewardV2DDPenaltyScale * currentDD;
   reward -= RewardV2DangerPenaltyScale * replayRisk;
   reward -= RewardV2MarginPenaltyScale * marginStress;
   reward -= RewardV2AgePenaltyPerDay * ageDays;

   reward -= RewardV2ArchiveCloseRiskPenaltyScale * archiveBadRisk;
   reward -= 0.60 * RewardV2ArchivePatternPenaltyScale * archivePatternPenalty;
   reward -= 0.60 * RewardV2ArchiveRegimePenaltyScale * archiveRegimePenalty;

   reward -= RewardV2ReplayCloseAntiPatternScale *
             (0.55 * replayDangerRisk + 0.95 * replayDeepRisk + RewardV2ReplayRecentCautionScale * MathMax(0.0, replayRecentCaution));

   if(closeRes.closedProfit > 0.0)
   {
      reward -= RewardV2RiskyProfitPenaltyScale *
                RewardV2RiskyProfitPenalty(MathAbs(profitRewardNorm), episodeAdds, episodeMaxPositions, episodeMaxDD,
                                           archiveDeepRisk + 0.35*replayDeepRisk, archiveBadRisk + 0.25*replayDangerRisk);

      if(episodeMaxPositions <= 1)
      {
         reward += RewardV2ArchiveCleanBonusScale * archiveCleanQuality;
         reward += RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 2.0);
         reward += 0.16 * recentCalm;
      }
      else
      {
         reward -= 0.35 * RewardV2ReplayCloseAntiPatternScale * Clamp(replayDeepRisk, 0.0, 3.0);
         reward -= 0.16 * recentDeepRate;
      }
   }

   if(closeRes.closedProfit <= 0.0 && episodeAdds > 0)
      reward -= RewardV2FailedBasketPenaltyScale * (1.0 + (double)episodeAdds);

   reward *= PendingCloseRewardScale;
   if(closeRes.closedProfit > 0.0 && episodeMaxPositions <= 1)
      reward += PendingGoodCloseBonus;

   reward = ScaleRewardByRegime(reward, extreme);
   return Clamp(reward, -RewardV2RewardClamp, RewardV2RewardClamp);
}




double ComputeOpenRewardV2(const int symIdx,
                           const bool isBuy,
                           const int positionsBeforeOpen,
                           const double combinedTrend,
                           const double reversalRisk,
                           const bool extreme)
{
   double reward = -RewardV2OpenBaseCost;

   int addCount = MathMax(positionsBeforeOpen, 0);
   int resultingPositions = addCount + 1;
   int basketDir = (isBuy ? 1 : -1);

   reward -= RewardV2QuadraticAddPenalty(addCount);
   reward -= 0.85 * RewardV2DeepBasketPenalty(resultingPositions);

   double replayRisk = CurrentReplayRiskBias(symIdx);
   reward -= RewardV2DangerPenaltyScale * replayRisk;

   double marginStress = CurrentMarginStressRatio();
   reward -= RewardV2MarginPenaltyScale * marginStress;

   double currentDD = GetSymbolFloatingDDPct(symIdx);
   reward -= 0.60 * RewardV2EpisodeMaxDDPenaltyScale * Clamp(currentDD, 0.0, 1.0);
   reward -= 0.50 * RewardV2DDPenaltyScale * Clamp(currentDD, 0.0, 1.0);

   double ageDays = BasketAgeDays(symIdx);
   reward -= 0.50 * RewardV2AgePenaltyPerDay * ageDays;
   if(addCount > 0)
      reward -= RewardV2AgingAddPenaltyScale * ageDays * (double)addCount;

   reward -= RewardV2ReversalPenaltyScale * Clamp(reversalRisk, 0.0, 1.0);

   double archiveBadRisk=0.0, archiveCleanQuality=0.0, archiveDeepRisk=0.0, archiveEfficiency=0.0;
   ComputeArchiveEpisodeProfile(symIdx, basketDir, resultingPositions, archiveBadRisk, archiveCleanQuality, archiveDeepRisk, archiveEfficiency);

   double archivePatternPenalty=0.0, archiveRegimePenalty=0.0;
   ComputeArchivePatternAndRegimePenalty(symIdx, archivePatternPenalty, archiveRegimePenalty);

   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double recentCalm=ComputeRecentDrawdownCalmBonus(symIdx);
   double periodicQualityTilt=Clamp((recentQuality - 0.5)*2.0,-1.0,1.0);

   reward -= RewardV2ArchiveOpenRiskPenaltyScale * (0.70 * archiveBadRisk + 0.30 * archiveDeepRisk);
   reward -= 0.50 * RewardV2ArchivePatternPenaltyScale * archivePatternPenalty;
   reward -= 0.50 * RewardV2ArchiveRegimePenaltyScale * archiveRegimePenalty;

   double replayDangerRisk=0.0, replayDeepRisk=0.0, replayEfficient=0.0, replayRecentCaution=0.0;
   ComputeReplayRewardProfile(symIdx, basketDir, resultingPositions, replayDangerRisk, replayDeepRisk, replayEfficient, replayRecentCaution);

   reward -= RewardV2ReplayOpenAntiPatternScale *
             (0.60 * replayDangerRisk + 0.90 * replayDeepRisk + RewardV2ReplayRecentCautionScale * MathMax(0.0, replayRecentCaution));

   if(addCount <= 0)
   {
      reward += RewardV2ArchiveCleanBonusScale * archiveCleanQuality;
      reward += 0.15 * RewardV2ArchiveCleanBonusScale * Clamp(archiveEfficiency, 0.0, 1.0);
      reward += RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 2.0);

      reward += 0.35 * periodicQualityTilt;
      reward += 0.18 * recentCalm;
      reward -= 0.30 * recentDeepRate;
   }
   else
   {
      reward -= 0.35 * RewardV2ReplayOpenAntiPatternScale * Clamp(replayDeepRisk, 0.0, 3.0);
      reward += 0.10 * RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 1.5);

      reward -= 0.18 * recentDeepRate;
      reward -= 0.12 * (1.0 - recentQuality);
   }

   reward *= PendingOpenRewardScale;
   reward = ScaleRewardByRegime(reward, extreme);
   return Clamp(reward, -RewardV2RewardClamp, RewardV2RewardClamp);
}



double GetEAOpenPnL()
{
   double totalPnL = 0.0;
   int total = PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      string sym = PositionGetString(POSITION_SYMBOL);
      long   mg  = PositionGetInteger(POSITION_MAGIC);
      int idx = SymbolIndex(sym);
      if(idx<0) continue;
      if((int)mg != gMagics[idx]) continue;
      totalPnL += PositionGetDouble(POSITION_PROFIT);
   }
   return totalPnL;
}

double GetEAEquity()
{
   return gEAStartEquity + gEAClosedProfit + GetEAOpenPnL();
}

void RefreshRewardTickBaseline()
{
   datetime now=TimeCurrent();
   if(gRewardBaselineTick==now && gTickBalanceBaseline>0.0) return;

   gTickEquityBaseline=GetEAEquity();
   gTickBalanceBaseline=gEAStartEquity;
   if(gTickBalanceBaseline<=1e-9) gTickBalanceBaseline=gTickEquityBaseline;
   if(gTickBalanceBaseline<=1e-9) gTickBalanceBaseline=1000.0;
   gRewardBaselineTick=now;
}

bool CheckEquityStop()
{
   if(!UseEquityStop) return false;
   if(EquityRiskPercent <= 0.0) return false;
   if(gAccountEquityStopPeak <= 0.0) return false;

   double eq = GetWatchedAccountEquity();
   return (eq < gAccountEquityStopPeak * (1.0 - EquityRiskPercent/100.0));
}

double GetCurrentEAFloatingLossMoney()
{
   double openPnL = GetEAOpenPnL();
   if(openPnL >= 0.0) return 0.0;
   return -openPnL;
}

bool CheckEquityLossStop()
{
   if(!UseEquityLossStop) return false;
   if(EquityLossStopAmount <= 0.0) return false;

   double floatingLoss = 0.0;

   if(UseGlobalAccountWatchdog)
   {
      double totalPnL = 0.0;

      for(int i=0; i<PositionsTotal(); i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         if(WatchAllAccountPositions)
         {
            totalPnL += PositionGetDouble(POSITION_PROFIT);
         }
         else
         {
            int mg = (int)PositionGetInteger(POSITION_MAGIC);
            if(mg >= WatchdogMagicMin && mg <= WatchdogMagicMax)
               totalPnL += PositionGetDouble(POSITION_PROFIT);
         }
      }

      if(totalPnL < 0.0)
         floatingLoss = -totalPnL;
   }
   else
   {
      floatingLoss = GetCurrentEAFloatingLossMoney();
   }

   return (floatingLoss >= EquityLossStopAmount);
}

void StartEquityLossStopCooldown()
{
   int sec = MathMax(0, EquityLossStopCooldownSeconds);
   if(sec <= 0)
      gEquityLossStopResumeTime = 0;
   else
      gEquityLossStopResumeTime = TimeCurrent() + sec;
}

void ResetEquityLossStopCooldownIfExpired()
{
   if(gEquityLossStopResumeTime > 0 && TimeCurrent() >= gEquityLossStopResumeTime)
      gEquityLossStopResumeTime = 0;
}

double GetWatchedAccountEquity()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double openPnL = 0.0;

   if(UseGlobalAccountWatchdog)
   {
      for(int i=0; i<PositionsTotal(); i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         if(WatchAllAccountPositions)
         {
            openPnL += PositionGetDouble(POSITION_PROFIT);
         }
         else
         {
            int mg = (int)PositionGetInteger(POSITION_MAGIC);
            if(mg >= WatchdogMagicMin && mg <= WatchdogMagicMax)
               openPnL += PositionGetDouble(POSITION_PROFIT);
         }
      }
   }
   else
   {
      // fallback to this EA's own live positions only
      for(int i=0;i<PositionsTotal();i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         string sym = PositionGetString(POSITION_SYMBOL);
         int idx = SymbolIndex(sym);
         if(idx<0) continue;

         int mg = (int)PositionGetInteger(POSITION_MAGIC);
         if(mg != gMagics[idx]) continue;

         openPnL += PositionGetDouble(POSITION_PROFIT);
      }
   }

   return balance + openPnL;
}


bool ProfitPauseActive()
{
   if(!UseProfitPause) return false;
   if(gProfitPauseResumeTime <= 0) return false;
   return (TimeCurrent() < gProfitPauseResumeTime);
}

void ResetProfitPauseIfExpired()
{
   if(gProfitPauseResumeTime > 0 && TimeCurrent() >= gProfitPauseResumeTime)
   {
      gProfitPauseResumeTime = 0;
      gProfitCycleClosedProfit = 0.0;
   }
}

void ResetForcedEntryState(const int symIdx)
{
   gForcedEntryActive[symIdx]   = false;
   gForcedEntryArmTime[symIdx]  = 0;
   gForcedEntryDeadline[symIdx] = 0;
}

void MarkTradeOpened(const int symIdx, const datetime when)
{
   gLastTradeOpenTime[symIdx] = (when>0 ? when : TimeCurrent());
   ResetForcedEntryState(symIdx);
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
      gDecisionSupportCache[symIdx].valid=false;
}

void UpdateForcedEntryWatchdog(const int symIdx)
{
   if(!UseForcedEntryWatchdog) return;

   datetime now = TimeCurrent();

   if(gLastTradeOpenTime[symIdx] <= 0)
      gLastTradeOpenTime[symIdx] = now;

   // if basket is already open, watchdog is not needed
   if(gPositionsCount[symIdx] > 0)
   {
      ResetForcedEntryState(symIdx);
      return;
   }

   int idleSec   = MathMax(1, ForcedEntryIdleMinutes) * 60;
   int windowSec = MathMax(1, ForcedEntryWindowMinutes) * 60;

   if(!gForcedEntryActive[symIdx])
   {
      if((now - gLastTradeOpenTime[symIdx]) >= idleSec)
      {
         gForcedEntryActive[symIdx]   = true;
         gForcedEntryArmTime[symIdx]  = now;
         gForcedEntryDeadline[symIdx] = now + windowSec;

         Print("FORCED ENTRY ARMED | sym=", gSymbols[symIdx],
               " idleMin=", ForcedEntryIdleMinutes,
               " deadline=", TimeToString(gForcedEntryDeadline[symIdx], TIME_DATE|TIME_SECONDS));
      }
   }
}

bool ForcedEntryRequiredNow(const int symIdx)
{
   if(!UseForcedEntryWatchdog) return false;
   if(!gForcedEntryActive[symIdx]) return false;
   if(gPositionsCount[symIdx] > 0) return false;
   return true;
}


int ArgMaxActionFromQ(const double &q[])
{
   int n=ArraySize(q);
   if(n<=0) return 0;

   int best=0;
   double bestV=q[0];
   for(int a=1;a<n;a++)
      if(q[a]>bestV){ bestV=q[a]; best=a; }

   return best;
}

int ArgMaxDirectionalActionFromQ(const double &q[])
{
   int n=ArraySize(q);
   if(n>=3) return (q[1] >= q[2] ? 1 : 2);
   if(n>=2) return 1;
   return 0;
}

double ComputeDDQNActionGap(const double &q[], const bool forcedDir=false)
{
   int n=ArraySize(q);
   if(n<=1) return 0.0;

   if(forcedDir && n>=3)
      return MathMax(0.0, MathAbs(q[1]-q[2]));

   double best=-1e100, second=-1e100;
   for(int a=0;a<n;a++)
   {
      double v=q[a];
      if(v>best)
      {
         second=best;
         best=v;
      }
      else if(v>second)
      {
         second=v;
      }
   }

   if(second<=-1e99) second=best;
   return MathMax(0.0, best-second);
}

double ComputeDrawdownWorseningScore(const int symIdx)
{
   double curDD = GetSymbolFloatingDDPct(symIdx);
   double prevDD = curDD;

   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gDenseBasketHealth[symIdx].active)
      prevDD = gDenseBasketHealth[symIdx].prevDD;

   double worsen = MathMax(0.0, curDD - prevDD);
   return Clamp(SafeDiv(worsen, 0.02, 0.0), 0.0, 1.0);
}

void InitActionBias(double &bias[])
{
   ArrayResize(bias, ActionCount);
   for(int a=0;a<ActionCount;a++) bias[a]=0.0;
}

void AccumulateActionBias(double &dst[], const double &src[])
{
   int n=MathMin(ArraySize(dst), ArraySize(src));
   for(int a=0;a<n;a++)
      dst[a] += src[a];
}

void BuildQMemoryVoteBias(const int symIdx,
                          const double &qBase[],
                          const double &qMem[],
                          const double memConf,
                          const bool forcedDir,
                          double &outBias[])
{
   InitActionBias(outBias);
   if(ArraySize(qBase)<=0 || ArraySize(qMem)<=0) return;

   double conf = Clamp(memConf, 0.0, 1.0);
   if(conf<=1e-12) return;

   double base = conf * QMemBlendWeight;
   double ddPct = GetSymbolFloatingDDPct(symIdx);
   double worsen = ComputeDrawdownWorseningScore(symIdx);
   double pDanger = Clamp(gPDanger[symIdx], 0.0, 1.0);

   int ddqnBest = ArgMaxActionFromQ(qBase);
   int qmemBest = ArgMaxActionFromQ(qMem);
   int ddqnDir  = ArgMaxDirectionalActionFromQ(qBase);
   int qmemDir  = ArgMaxDirectionalActionFromQ(qMem);

   double qMemGap = ComputeDDQNActionGap(qMem, forcedDir);
   double intensity = 0.60 + 0.40 * Clamp(SafeDiv(qMemGap, 1.0, 0.0), 0.0, 1.0);

   if(!forcedDir && ActionCount>=3)
   {
      if(qmemBest==ddqnBest)
      {
         double support = base * QMemConfirmScale * intensity * (0.75 + 0.25 * (1.0 - ddPct));
         outBias[ddqnBest] += support;
         if(ddqnBest==0)
            outBias[0] += 0.15 * support;
      }
      else
      {
         double caution = base * QMemConflictToHoldScale * intensity * (0.70 + 0.80*ddPct + 0.40*worsen + 0.35*pDanger);
         outBias[0] += caution;
         if(ddqnBest>=1 && ddqnBest<=2)
            outBias[ddqnBest] -= caution * QMemConflictDirPenaltyScale;
      }
   }
   else
   {
      if(qmemDir==ddqnDir)
      {
         double support = base * QMemConfirmScale * 0.60 * intensity;
         if(ddqnDir>=1 && ddqnDir<=2)
            outBias[ddqnDir] += support;
      }
      else
      {
         double caution = base * 0.60 * intensity * (0.60 + ddPct + 0.40*worsen + 0.35*pDanger);
         if(ddqnDir>=1 && ddqnDir<=2)
            outBias[ddqnDir] -= caution * QMemConflictDirPenaltyScale;

         if(qmemDir>=1 && qmemDir<=2 && qmemDir!=ddqnDir)
            outBias[qmemDir] += 0.15 * caution;
      }
   }
}

void BuildDDEventCautionBias(const int symIdx,
                             const double &qBase[],
                             const double &ddBiasIn[],
                             const bool forcedDir,
                             double &outBias[])
{
   InitActionBias(outBias);
   if(ArraySize(ddBiasIn)<=0 || ArraySize(qBase)<=0) return;

   double ddPct = GetSymbolFloatingDDPct(symIdx);
   double worsen = ComputeDrawdownWorseningScore(symIdx);
   double pDanger = Clamp(gPDanger[symIdx], 0.0, 1.0);

   double boost = DDEventBlendWeight * (1.0 + DDEventDrawdownBoostScale * (0.80*ddPct + 0.70*worsen + 0.30*pDanger));

   int ddqnBest = ArgMaxActionFromQ(qBase);
   int ddqnDir  = ArgMaxDirectionalActionFromQ(qBase);

   if(!forcedDir && ActionCount>=3)
   {
      outBias[0] += MathMax(0.0, ddBiasIn[0]) * boost;
      if(ddqnBest>=1 && ddqnBest<=2)
      {
         double dirPenalty = MathMax(0.0, -MathMin(0.0, ddBiasIn[ddqnBest])) * boost;
         outBias[ddqnBest] -= dirPenalty;
      }
   }
   else
   {
      if(ddqnDir>=1 && ddqnDir<=2)
      {
         double dirPenalty = MathMax(0.0, -MathMin(0.0, ddBiasIn[ddqnDir])) * boost;
         outBias[ddqnDir] -= dirPenalty;

         if(!DDEventUseCautionOnly)
         {
            int alt=(ddqnDir==1 ? 2 : 1);
            if(alt>=1 && alt<=2 && alt<ArraySize(ddBiasIn))
               outBias[alt] += 0.15 * MathMax(0.0, ddBiasIn[alt]) * boost;
         }
      }
   }
}

void BuildDangerVetoBias(const int symIdx,
                         const double &qBase[],
                         const double &dangerBiasIn[],
                         const double simBest,
                         const bool forcedDir,
                         double &outBias[])
{
   InitActionBias(outBias);
   if(ArraySize(dangerBiasIn)<=0 || ArraySize(qBase)<=0) return;

   double alpha = ComputeAlphaMix(symIdx, simBest);
   if(alpha<=1e-12) return;

   double ddPct = GetSymbolFloatingDDPct(symIdx);
   double worsen = ComputeDrawdownWorseningScore(symIdx);
   double pDanger = Clamp(gPDanger[symIdx], 0.0, 1.0);

   double severity = alpha * (0.75 + 0.65*pDanger + 0.75*ddPct + 0.60*worsen);
   if(gMode[symIdx]==MODE_DANGER)  severity *= 1.15;
   if(gMode[symIdx]==MODE_CAUTION) severity *= 0.90;

   int ddqnBest = ArgMaxActionFromQ(qBase);
   int ddqnDir  = ArgMaxDirectionalActionFromQ(qBase);

   if(!forcedDir && ActionCount>=3)
   {
      double holdVeto = DangerHoldVetoScale * severity * (0.40 + MathMax(0.0, dangerBiasIn[0]));
      outBias[0] += holdVeto;

      if(ddqnBest>=1 && ddqnBest<=2)
      {
         double dirPenalty = DangerHoldVetoScale * severity * (0.30 + MathMax(0.0, -MathMin(0.0, dangerBiasIn[ddqnBest])));
         outBias[ddqnBest] -= dirPenalty;

         if(gMode[symIdx]==MODE_CAUTION && dangerBiasIn[ddqnBest] > 0.0)
            outBias[ddqnBest] += DangerDirectionalLeakScale * severity * 0.25 * dangerBiasIn[ddqnBest];
      }
   }
   else
   {
      if(ddqnDir>=1 && ddqnDir<=2)
      {
         double dirPenalty = DangerHoldVetoScale * severity * (0.25 + MathMax(0.0, -MathMin(0.0, dangerBiasIn[ddqnDir])));
         outBias[ddqnDir] -= dirPenalty;

         int alt=(ddqnDir==1 ? 2 : 1);
         if(alt>=1 && alt<=2 && alt<ArraySize(dangerBiasIn) && dangerBiasIn[alt] > 0.0)
            outBias[alt] += DangerDirectionalLeakScale * severity * dangerBiasIn[alt];
      }
   }
}

void ApplyBudgetedSubordinateBias(const int symIdx,
                                  const bool forcedDir,
                                  const double &qBase[],
                                  const double &delta[],
                                  double &qOut[])
{
   int n=ArraySize(qBase);
   ArrayResize(qOut,n);
   for(int a=0;a<n;a++) qOut[a]=qBase[a];

   double gap = ComputeDDQNActionGap(qBase, forcedDir);
   double budget = Clamp(SubordinateBiasCapFrac, 0.0, 1.0) * gap;
   if(budget<=1e-12) return;

   double maxAbs=0.0;
   int start = (forcedDir && n>=3 ? 1 : 0);
   int end   = (forcedDir && n>=3 ? 3 : n);

   for(int a=start;a<end;a++)
      maxAbs = MathMax(maxAbs, MathAbs(delta[a]));

   if(maxAbs<=1e-12) return;

   double scale = MathMin(1.0, SafeDiv(budget, maxAbs, 1.0));
   for(int a=0;a<n;a++)
      qOut[a] += scale * delta[a];
}



double MaxAbsArrayValue(const double &arr[])
{
   double out=0.0;
   int n=ArraySize(arr);
   for(int i=0;i<n;i++)
      out=MathMax(out,MathAbs(arr[i]));
   return out;
}

void ResetDecisionSupportContext(DecisionSupportContext &ctx)
{
   ctx.valid=false;
   ctx.forcedEntry=false;
   ctx.barTime=0;
   ctx.refreshTime=0;
   ctx.regime=0;
   ctx.positionsCount=0;
   ctx.basketDir=0;
   ctx.brainMode=0;
   for(int a=0;a<3;a++) ctx.actionDelta[a]=0.0;
   ctx.qMemoryAgreement=0.0;
   ctx.qMemoryConflict=0.0;
   ctx.archiveCaution=0.0;
   ctx.ddEventRisk=0.0;
   ctx.dangerProbability=0.0;
   ctx.oneRoundPrior=0.0;
   ctx.addRiskPrior=0.0;
   ctx.supportConfidence=0.0;
   ctx.painRecurrenceRisk=0.0;
   ctx.macroReversalTrapPrior=0.0;
   ctx.counterTrendFailurePrior=0.0;
   ctx.regimeBreakPainPrior=0.0;
   ctx.recoveryFalseStartRisk=0.0;
   ctx.deepBasketPainPrior=0.0;
   ctx.painMemoryAgreement=0.0;
   ctx.macroMicroConflict=0.0;
   ctx.lateTrendFadePenalty=0.0;
   ctx.painConfidence=0.0;
   ctx.trendPersistenceProb=0.0;
   ctx.trendReversalProb=0.0;
   ctx.spikeRiskProb=0.0;
   ctx.expectedBasketDepth=0.0;
   ctx.trendContinuationQuality=0.0;
   ctx.breakoutReclaimQuality=0.0;
   ctx.reversalTransitionQuality=0.0;
   ctx.modeDominanceScore=0.0;
   ctx.modeConflictScore=0.0;
}

void CopyDecisionSupportContext(const DecisionSupportContext &src, DecisionSupportContext &dst)
{
   dst.valid=src.valid;
   dst.forcedEntry=src.forcedEntry;
   dst.barTime=src.barTime;
   dst.refreshTime=src.refreshTime;
   dst.regime=src.regime;
   dst.positionsCount=src.positionsCount;
   dst.basketDir=src.basketDir;
   dst.brainMode=src.brainMode;
   for(int a=0;a<3;a++) dst.actionDelta[a]=src.actionDelta[a];
   dst.qMemoryAgreement=src.qMemoryAgreement;
   dst.qMemoryConflict=src.qMemoryConflict;
   dst.archiveCaution=src.archiveCaution;
   dst.ddEventRisk=src.ddEventRisk;
   dst.dangerProbability=src.dangerProbability;
   dst.oneRoundPrior=src.oneRoundPrior;
   dst.addRiskPrior=src.addRiskPrior;
   dst.supportConfidence=src.supportConfidence;
   dst.painRecurrenceRisk=src.painRecurrenceRisk;
   dst.macroReversalTrapPrior=src.macroReversalTrapPrior;
   dst.counterTrendFailurePrior=src.counterTrendFailurePrior;
   dst.regimeBreakPainPrior=src.regimeBreakPainPrior;
   dst.recoveryFalseStartRisk=src.recoveryFalseStartRisk;
   dst.deepBasketPainPrior=src.deepBasketPainPrior;
   dst.painMemoryAgreement=src.painMemoryAgreement;
   dst.macroMicroConflict=src.macroMicroConflict;
   dst.lateTrendFadePenalty=src.lateTrendFadePenalty;
   dst.painConfidence=src.painConfidence;
   dst.trendPersistenceProb=src.trendPersistenceProb;
   dst.trendReversalProb=src.trendReversalProb;
   dst.spikeRiskProb=src.spikeRiskProb;
   dst.expectedBasketDepth=src.expectedBasketDepth;
   dst.trendContinuationQuality=src.trendContinuationQuality;
   dst.breakoutReclaimQuality=src.breakoutReclaimQuality;
   dst.reversalTransitionQuality=src.reversalTransitionQuality;
   dst.modeDominanceScore=src.modeDominanceScore;
   dst.modeConflictScore=src.modeConflictScore;
}

datetime DecisionSupportBarTime(const string symbol)
{
   datetime barTime=iTime(symbol,BaseTF,1);
   if(barTime<=0) barTime=iTime(symbol,BaseTF,0);
   if(barTime<=0) barTime=TimeCurrent();
   return barTime;
}

void ResetDecisionSupportCacheAll()
{
   for(int i=0;i<MAX_SYMBOLS;i++)
      ResetDecisionSupportContext(gDecisionSupportCache[i]);
}

void ApplyDecisionSupportDelta(const DecisionSupportContext &ctx,double &totalDelta[])
{
   if(ArraySize(totalDelta)<ActionCount)
      ArrayResize(totalDelta,ActionCount);
   int n=MathMin(ActionCount,3);
   for(int a=0;a<n;a++)
      totalDelta[a]+=ctx.actionDelta[a];
}

bool BuildDecisionSupportContext(const int symIdx,
                                 const int regime,
                                 const double &state[],
                                 const double &qBase[],
                                 const bool forcedEntry,
                                 const int basketDirHint,
                                 DecisionSupportContext &ctx)
{
   ResetDecisionSupportContext(ctx);
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return false;

   string symbol=gSymbols[symIdx];
   datetime barTime=DecisionSupportBarTime(symbol);
   int positionsCount=gPositionsCount[symIdx];
   int basketDir=basketDirHint;
   if(positionsCount>0 && basketDir==0)
      basketDir=BasketDir(symbol,gMagics[symIdx]);
   int brainMode=(int)gMode[symIdx];

   bool cacheOk=(UseDecisionSupportCache &&
                 gDecisionSupportCache[symIdx].valid &&
                 gDecisionSupportCache[symIdx].forcedEntry==forcedEntry &&
                 gDecisionSupportCache[symIdx].barTime==barTime &&
                 gDecisionSupportCache[symIdx].regime==regime &&
                 gDecisionSupportCache[symIdx].positionsCount==positionsCount &&
                 gDecisionSupportCache[symIdx].basketDir==basketDir &&
                 gDecisionSupportCache[symIdx].brainMode==brainMode);

   if(cacheOk)
   {
      CopyDecisionSupportContext(gDecisionSupportCache[symIdx],ctx);
      return true;
   }

   ctx.valid=true;
   ctx.forcedEntry=forcedEntry;
   ctx.barTime=barTime;
   ctx.refreshTime=TimeCurrent();
   ctx.regime=regime;
   ctx.positionsCount=positionsCount;
   ctx.basketDir=basketDir;
   ctx.brainMode=brainMode;
   ctx.dangerProbability=Clamp(gPDanger[symIdx],0.0,1.0);

   double totalDelta[];
   InitActionBias(totalDelta);

   double supportAcc=0.0;
   double supportW=0.0;

   double stateKey[];
   bool haveStateKey=BuildQMemoryKey(symIdx,state,stateKey);

   if(UseQMemory && haveStateKey)
   {
      double qMem[];
      double memConf=0.0;
      if(GetQMemoryForDecision(symIdx,regime,symbol,stateKey,qMem,memConf))
      {
         double qBias[];
         BuildQMemoryVoteBias(symIdx,qBase,qMem,memConf,forcedEntry,qBias);
         AccumulateActionBias(totalDelta,qBias);

         ctx.qMemoryAgreement=ArchiveLiveDecisionConfidence(qMem);
         ctx.qMemoryConflict=1.0-ctx.qMemoryAgreement;
         supportAcc += Clamp(0.60*memConf + 0.40*ctx.qMemoryAgreement,0.0,1.0);
         supportW   += 1.0;
      }
   }

   if(UseDDEventBias && haveStateKey)
   {
      double ddBiasRaw[];
      if(QueryDDEventBiasWithKey(symIdx, regime, -1, basketDir, stateKey, ddBiasRaw))
      {
         double ddBias[];
         BuildDDEventCautionBias(symIdx,qBase,ddBiasRaw,forcedEntry,ddBias);
         AccumulateActionBias(totalDelta,ddBias);

         double holdCaution=(ArraySize(ddBiasRaw)>0 ? MathMax(0.0,ddBiasRaw[0]) : 0.0);
         double longPenalty=(ArraySize(ddBiasRaw)>1 ? MathMax(0.0,-ddBiasRaw[1]) : 0.0);
         double shortPenalty=(ArraySize(ddBiasRaw)>2 ? MathMax(0.0,-ddBiasRaw[2]) : 0.0);
         ctx.ddEventRisk=Clamp(0.50*holdCaution + 0.25*longPenalty + 0.25*shortPenalty,0.0,1.0);

         supportAcc += Clamp(MaxAbsArrayValue(ddBiasRaw),0.0,1.0);
         supportW   += 1.0;
      }
   }

   if(UseArchiveLiveSimilarity)
   {
      double archiveBias[];
      double archiveAlpha=0.0;
      if(BuildArchiveLiveActionBias(symIdx, regime, state, qBase, archiveBias, archiveAlpha))
      {
         double scaledArchive[];
         InitActionBias(scaledArchive);
         int nA=MathMin(ArraySize(scaledArchive),ArraySize(archiveBias));
         for(int a=0;a<nA;a++)
            scaledArchive[a]=archiveAlpha*archiveBias[a];
         AccumulateActionBias(totalDelta,scaledArchive);

         double holdBias=(ArraySize(archiveBias)>0 ? archiveBias[0] : 0.0);
         double dirBias=0.0;
         if(ArraySize(archiveBias)>2)
            dirBias=MathMax(archiveBias[1],archiveBias[2]);

         ctx.archiveCaution=Clamp(MathMax(0.0,archiveAlpha*holdBias),0.0,1.0);
         ctx.oneRoundPrior =Clamp(MathMax(0.0,archiveAlpha*dirBias),0.0,1.0);

         supportAcc += Clamp(archiveAlpha,0.0,1.0);
         supportW   += 1.0;
      }
   }

   if(UseDangerBrain && gMode[symIdx]!=MODE_NORMAL)
   {
      double f_now[];
      ArrayResize(f_now,6);
      for(int k=0;k<6;k++) f_now[k]=gFPCache[symIdx][k];

      double rawBias[];
      double simBest=-1e9;
      if(GetBestAdapterBiasSmart(symIdx,f_now,rawBias,simBest))
      {
         double dangerBias[];
         BuildDangerVetoBias(symIdx,qBase,rawBias,simBest,forcedEntry,dangerBias);
         AccumulateActionBias(totalDelta,dangerBias);

         double dangerConf=Clamp((simBest+1.0)*0.5,0.0,1.0);
         ctx.dangerProbability=Clamp(MathMax(ctx.dangerProbability,dangerConf),0.0,1.0);

         supportAcc += dangerConf;
         supportW   += 1.0;
      }
   }

   if(haveStateKey)
   {
      ComputePainMemorySupport(symIdx,
                               regime,
                               symbol,
                               stateKey,
                               ctx.painRecurrenceRisk,
                               ctx.macroReversalTrapPrior,
                               ctx.counterTrendFailurePrior,
                               ctx.regimeBreakPainPrior,
                               ctx.recoveryFalseStartRisk,
                               ctx.deepBasketPainPrior,
                               ctx.painMemoryAgreement,
                               ctx.macroMicroConflict,
                               ctx.lateTrendFadePenalty,
                               ctx.painConfidence);

      supportAcc += Clamp(0.45*ctx.painConfidence +
                          0.30*ctx.painRecurrenceRisk +
                          0.25*ctx.painMemoryAgreement,0.0,1.0);
      supportW   += 1.0;
   }

   ctx.addRiskPrior=Clamp(0.22*ctx.archiveCaution +
                          0.18*ctx.ddEventRisk +
                          0.18*ctx.dangerProbability +
                          0.14*CurrentReplayRiskBias(symIdx) +
                          0.10*ctx.painRecurrenceRisk +
                          0.08*ctx.deepBasketPainPrior +
                          0.05*ctx.macroReversalTrapPrior +
                          0.05*ctx.regimeBreakPainPrior,0.0,1.0);
   ctx.oneRoundPrior=Clamp(0.82*ctx.oneRoundPrior +
                           0.10*(1.0-ctx.deepBasketPainPrior) +
                           0.08*(1.0-ctx.macroReversalTrapPrior),0.0,1.0);
   ctx.supportConfidence=(supportW>0.0 ? Clamp(supportAcc/supportW,0.0,1.0) : 0.0);

   for(int a=0;a<3;a++)
      ctx.actionDelta[a]=(a<ArraySize(totalDelta) ? totalDelta[a] : 0.0);

   ctx.trendPersistenceProb = ComputeOptionBTrendPersistenceProb(symIdx,state,qBase,ctx);
   ctx.trendReversalProb    = ComputeOptionBTrendReversalProb(symIdx,state,qBase,ctx);
   ctx.spikeRiskProb        = ComputeOptionBSpikeRiskProb(symIdx,state,qBase,ctx);
   ctx.expectedBasketDepth  = ComputeOptionBExpectedBasketDepth(symIdx,state,qBase,ctx,
                                                                ctx.trendPersistenceProb,
                                                                ctx.trendReversalProb,
                                                                ctx.spikeRiskProb);
   ComputeStrategyModeScores(symIdx,state,ctx,
                             ctx.trendContinuationQuality,
                             ctx.breakoutReclaimQuality,
                             ctx.reversalTransitionQuality,
                             ctx.modeDominanceScore,
                             ctx.modeConflictScore);

   if(UseDecisionSupportCache)
      CopyDecisionSupportContext(ctx,gDecisionSupportCache[symIdx]);

   return true;
}


double ComputeRecentDeepBasketRate(const int symIdx)
{
   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return 0.0;

   int used=0;
   double acc=0.0, wsum=0.0;
   for(int i=n-1; i>=0 && used<12; --i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      double w=1.0/(1.0 + 0.18*(double)used);
      double deep=(gEpisodeMemory[i].openPositionsMax >= DeepBasketAddThreshold ? 1.0 : 0.0);
      if(gEpisodeMemory[i].maxDrawdownPct >= DangerReplayDDThresholdPct) deep=MathMax(deep,0.80);
      if(gEpisodeMemory[i].inefficientRecovery>0) deep=MathMax(deep,0.65);
      acc += w * Clamp(deep,0.0,1.0);
      wsum += w;
      used++;
   }
   if(wsum<=0.0) return 0.0;
   return Clamp(acc/wsum,0.0,1.0);
}

double ComputeRecentDrawdownCalmBonus(const int symIdx)
{
   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return 0.5;

   int used=0;
   double acc=0.0, wsum=0.0;
   for(int i=n-1; i>=0 && used<12; --i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      double w=1.0/(1.0 + 0.16*(double)used);
      double calm=1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct, MathMax(RewardV2CleanCycleMaxDD,1e-6), 1.0);
      if(gEpisodeMemory[i].openPositionsMax >= DeepBasketAddThreshold) calm -= 0.35;
      if(gEpisodeMemory[i].forcedStopLikeEvent>0) calm -= 0.25;
      acc += w * Clamp(calm,-1.0,1.0);
      wsum += w;
      used++;
   }
   if(wsum<=0.0) return 0.5;
   return Clamp(0.5 + 0.5*(acc/wsum),0.0,1.0);
}

double ComputeRecentTradingQualityScore(const int symIdx)
{
   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return 0.5;

   int used=0;
   double acc=0.0, wsum=0.0;
   for(int i=n-1; i>=0 && used<12; --i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      double w=1.0/(1.0 + 0.15*(double)used);

      double score=0.0;
      if(gEpisodeMemory[i].pnlFinal > 0.0) score += 0.22;
      else if(gEpisodeMemory[i].pnlFinal < 0.0) score -= 0.18;

      score += 0.24 * Clamp(gEpisodeMemory[i].rewardEfficiency,-1.0,1.5);
      score += 0.18 * (gEpisodeMemory[i].oneRoundTrade>0 ? 1.0 : 0.0);
      score += 0.14 * Clamp(1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct, MathMax(RewardV2CleanCycleMaxDD,1e-6), 1.0), -1.0, 1.0);

      if(gEpisodeMemory[i].openPositionsMax >= DeepBasketAddThreshold) score -= 0.28;
      if(gEpisodeMemory[i].inefficientRecovery > 0) score -= 0.20;
      if(gEpisodeMemory[i].forcedStopLikeEvent > 0) score -= 0.25;
      if(gEpisodeMemory[i].maxDrawdownPct >= DangerReplayDDThresholdPct) score -= 0.22;

      acc += w * Clamp(score,-1.0,1.0);
      wsum += w;
      used++;
   }

   if(wsum<=0.0) return 0.5;
   return Clamp(0.5 + 0.5*(acc/wsum),0.0,1.0);
}


double ComputeCounterTrendTrapRiskScore(const string symbol,const int symIdx)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volE=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volM=0.0;
   double dirL=0.0,strL=0.0,persL=0.0,accL=0.0,overL=0.0,mrL=0.0,contL=0.0,revL=0.0,volL=0.0;

   ComputeTrendFactorTF(symbol, TF_EXEC, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volE);
   ComputeTrendFactorTF(symbol, TF_MID,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volM);
   ComputeTrendFactorTF(symbol, TF_LONG, point, dirL,strL,persL,accL,overL,mrL,contL,revL,volL);

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double setupConsensus = 1.0 - Clamp((MathAbs(dirE-dirM) + MathAbs(dirM-dirL) + MathAbs(dirE-dirL))/6.0, 0.0, 1.0);
   double bridgeAgreement = 1.0 - Clamp((MathAbs(dirM-dirL) + MathAbs(dirL-macroBias))/4.0, 0.0, 1.0);
   double continuation = Clamp(0.42*setupConsensus*(0.50*contE + 0.30*contM + 0.20*contL) + 0.38*macroContinuation + 0.20*bridgeAgreement, 0.0, 1.0);
   double reversal = Clamp(0.30*(0.45*revE + 0.35*revM + 0.20*revL) + 0.30*macroTransition + 0.20*MathAbs(macroReclaim) + 0.20*(1.0-macroContinuation), 0.0, 1.0);
   double maturity = Clamp(0.40*(0.55*overE + 0.30*overM + 0.15*overL) + 0.40*macroMaturity + 0.20*lateTrendTrap, 0.0, 1.0);

   return Clamp(0.34*continuation +
                0.16*(1.0-reversal) +
                0.12*maturity +
                0.12*(1.0-bridgeAgreement) +
                0.14*macroContinuation +
                0.07*MathAbs(macroReclaim) +
                0.05*ComputeRecentDeepBasketRate(symIdx), 0.0, 1.0);
}

double ComputeReversalConfirmationScore(const string symbol)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volE=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volM=0.0;
   double dirL=0.0,strL=0.0,persL=0.0,accL=0.0,overL=0.0,mrL=0.0,contL=0.0,revL=0.0,volL=0.0;

   ComputeTrendFactorTF(symbol, TF_EXEC, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volE);
   ComputeTrendFactorTF(symbol, TF_MID,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volM);
   ComputeTrendFactorTF(symbol, TF_LONG, point, dirL,strL,persL,accL,overL,mrL,contL,revL,volL);

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double setupConsensus = 1.0 - Clamp((MathAbs(dirE-dirM) + MathAbs(dirM-dirL) + MathAbs(dirE-dirL))/6.0, 0.0, 1.0);
   double bridgeAgreement = 1.0 - Clamp((MathAbs(dirM-dirL) + MathAbs(dirL-macroBias))/4.0, 0.0, 1.0);

   return Clamp(0.24*(0.45*revE + 0.35*revM + 0.20*revL) +
                0.24*macroTransition +
                0.18*MathAbs(macroReclaim) +
                0.12*bridgeAgreement +
                0.12*(1.0-macroContinuation) +
                0.10*(1.0-lateTrendTrap), 0.0, 1.0);
}

double ComputeRegimeBreakScore(const string symbol,const int symIdx)
{
   double volExec = Clamp(0.55*RealizedVolLevelTF(symbol, TF_EXEC, 8) + 0.45*RealizedVolLevelTF(symbol, TF_EXEC, 24), 0.0, 1.0);
   double expExec = Clamp(0.45*RealizedVolDeltaTF(symbol, TF_EXEC, 8) + 0.25*RealizedVolGammaTF(symbol, TF_EXEC, 8) + 0.30*RangeExpansionTF(symbol, TF_EXEC, 5, 20), 0.0, 1.0);
   double expMid  = Clamp(0.45*RealizedVolDeltaTF(symbol, TF_MID, 8)  + 0.25*RealizedVolGammaTF(symbol, TF_MID, 8)  + 0.30*RangeExpansionTF(symbol, TF_MID, 5, 20), 0.0, 1.0);
   double wickInst = Clamp(0.50*WickInstabilityTF(symbol,TF_EXEC,1) + 0.30*WickInstabilityTF(symbol,TF_MID,1) + 0.20*WickInstabilityTF(symbol,TF_LONG,1), 0.0, 1.0);
   double spread = ComputeSpreadPressureScore(symbol,symIdx);
   return Clamp(0.40*expExec + 0.20*MathMax(0.0, expExec-volExec) + 0.15*MathMax(0.0, expMid-volExec) + 0.15*wickInst + 0.10*spread, 0.0, 1.0);
}

enum PainEventType
{
   PAIN_EVENT_GENERIC = 0,
   PAIN_EVENT_COUNTERTREND_TRAP = 1,
   PAIN_EVENT_LATE_TREND_FADE = 2,
   PAIN_EVENT_REGIME_BREAK = 3,
   PAIN_EVENT_FALSE_REVERSAL = 4,
   PAIN_EVENT_RECOVERY_FALSE_START = 5
};

void BuildMacroPainSignature(const string symbol,const int symIdx,double &sig[])
{
   ArrayResize(sig,6);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double h4Vol=Clamp(VolumeRatioTF(symbol, PERIOD_H4, 10),0.0,2.0)/2.0;

   sig[0]=Clamp(0.5 + 0.5*macroBias,0.0,1.0);
   sig[1]=Clamp(macroContinuation,0.0,1.0);
   sig[2]=Clamp(macroMaturity,0.0,1.0);
   sig[3]=Clamp(MathAbs(macroReclaim),0.0,1.0);
   sig[4]=Clamp(macroTransition,0.0,1.0);
   sig[5]=Clamp(0.70*regimeBreak + 0.30*h4Vol,0.0,1.0);
   NormalizeVecN(sig);
}

void BuildMicroPainSignature(const string symbol,const int symIdx,double &sig[])
{
   ArrayResize(sig,6);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double reversal=ComputeReversalConfirmationScore(symbol);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double spread=ComputeSpreadPressureScore(symbol,symIdx);
   double wick=Clamp(0.55*WickInstabilityTF(symbol,TF_EXEC,1)+0.45*WickInstabilityTF(symbol,TF_MID,1),0.0,1.0);
   double volBurst=Clamp(0.65*VolumeRatioTF(symbol,TF_EXEC,10)+0.35*VolumeRatioTF(symbol,TF_MID,10),0.0,2.0)/2.0;

   sig[0]=Clamp(trapRisk,0.0,1.0);
   sig[1]=Clamp(reversal,0.0,1.0);
   sig[2]=Clamp(regimeBreak,0.0,1.0);
   sig[3]=Clamp(spread,0.0,1.0);
   sig[4]=Clamp(wick,0.0,1.0);
   sig[5]=Clamp(volBurst,0.0,1.0);
   NormalizeVecN(sig);
}
void ComputePainMemorySupport(const int symIdx,
                              const int regime,
                              const string symbol,
                              const double &stateKey[],
                              double &painRecurrence,
                              double &macroTrapPrior,
                              double &counterTrendPrior,
                              double &regimeBreakPainPrior,
                              double &recoveryFalseStartRisk,
                              double &deepBasketPainPrior,
                              double &painAgreement,
                              double &macroMicroConflict,
                              double &lateTrendFadePenalty,
                              double &painConfidence)
{
   painRecurrence=0.0;
   macroTrapPrior=0.0;
   counterTrendPrior=0.0;
   regimeBreakPainPrior=0.0;
   recoveryFalseStartRisk=0.0;
   deepBasketPainPrior=0.0;
   painAgreement=0.0;
   macroMicroConflict=0.0;
   lateTrendFadePenalty=0.0;
   painConfidence=0.0;

   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   double curMacro[]; BuildMacroPainSignature(symbol,symIdx,curMacro);
   double curMicro[]; BuildMicroPainSignature(symbol,symIdx,curMicro);

   double macroSum=0.0, microSum=0.0, wsum=0.0;
   int start=MathMax(0,ArraySize(gDDEvents)-40);
   for(int i=start;i<ArraySize(gDDEvents);i++)
   {
      if(gDDEvents[i].symbol!=symbol) continue;
      if(UseRegimeBank && gDDEvents[i].regimeAtTrigger!=regime) continue;

      double simKey=0.0;
      if(ArraySize(stateKey)>0 && ArraySize(gDDEvents[i].triggerStateKey)==ArraySize(stateKey))
         simKey=Clamp(CosSim(stateKey,gDDEvents[i].triggerStateKey),0.0,1.0);
      double simMacro=(ArraySize(gDDEvents[i].macroSig)==ArraySize(curMacro) ? Clamp(CosSim(curMacro,gDDEvents[i].macroSig),0.0,1.0) : 0.0);
      double simMicro=(ArraySize(gDDEvents[i].microSig)==ArraySize(curMicro) ? Clamp(CosSim(curMicro,gDDEvents[i].microSig),0.0,1.0) : 0.0);
      double pain=Clamp(gDDEvents[i].painSeverity,0.0,1.0);
      double w=(0.30*simKey + 0.35*simMacro + 0.35*simMicro) * (1.0 + 0.90*pain);
      if(w<=1e-8) continue;

      painRecurrence += w * pain;
      deepBasketPainPrior += w * Clamp((double)gDDEvents[i].basketDepthMax / MathMax(3.0,(double)DeepBasketAddThreshold),0.0,1.0);
      double evMacroRegime=(ArraySize(gDDEvents[i].macroSig)>5 ? Clamp(gDDEvents[i].macroSig[5],0.0,1.0) : 0.0);
      regimeBreakPainPrior += w * (gDDEvents[i].eventType==PAIN_EVENT_REGIME_BREAK ? 1.0 : evMacroRegime);
      double evMicroTrap=(ArraySize(gDDEvents[i].microSig)>0 ? Clamp(gDDEvents[i].microSig[0],0.0,1.0) : 0.0);
      counterTrendPrior += w * (gDDEvents[i].eventType==PAIN_EVENT_COUNTERTREND_TRAP ? 1.0 : evMicroTrap);
      recoveryFalseStartRisk += w * (gDDEvents[i].eventType==PAIN_EVENT_RECOVERY_FALSE_START ? 1.0 : Clamp(gDDEvents[i].recoveryFailureScore,0.0,1.0));
      double evMacroMaturity=(ArraySize(gDDEvents[i].macroSig)>2 ? Clamp(gDDEvents[i].macroSig[2],0.0,1.0) : 0.0);
      lateTrendFadePenalty += w * (gDDEvents[i].eventType==PAIN_EVENT_LATE_TREND_FADE ? 1.0 : evMacroMaturity);
      double evMacroTransition=(ArraySize(gDDEvents[i].macroSig)>4 ? Clamp(gDDEvents[i].macroSig[4],0.0,1.0) : 0.0);
      double evMacroReclaim=(ArraySize(gDDEvents[i].macroSig)>3 ? Clamp(gDDEvents[i].macroSig[3],0.0,1.0) : 0.0);
      macroTrapPrior += w * Clamp(0.50*simMacro + 0.25*evMacroTransition + 0.25*evMacroReclaim,0.0,1.0);
      macroSum += w*simMacro;
      microSum += w*simMicro;
      wsum += w;
   }

   for(int i=0;i<ArraySize(gProtos);i++)
   {
      if(!gProtos[i].isDanger) continue;
      double simMacro=(ArraySize(gProtos[i].macroSig)==ArraySize(curMacro) ? Clamp(CosSim(curMacro,gProtos[i].macroSig),0.0,1.0) : 0.0);
      double simMicro=(ArraySize(gProtos[i].microSig)==ArraySize(curMicro) ? Clamp(CosSim(curMicro,gProtos[i].microSig),0.0,1.0) : 0.0);
      double pain=Clamp(gProtos[i].painMean,0.0,1.0);
      double w=(0.55*simMacro + 0.45*simMicro) * (1.0 + 0.75*pain);
      if(w<=1e-8) continue;

      painRecurrence += w * pain;
      deepBasketPainPrior += w * Clamp(gProtos[i].deepBasketRate,0.0,1.0);
      regimeBreakPainPrior += w * Clamp(gProtos[i].regimeBreakRate,0.0,1.0);
      counterTrendPrior += w * Clamp(gProtos[i].counterTrendFailureRate,0.0,1.0);
      recoveryFalseStartRisk += w * Clamp(gProtos[i].recoveryFailureRate,0.0,1.0);
      lateTrendFadePenalty += w * Clamp(gProtos[i].reversalTrapRate,0.0,1.0);
      macroTrapPrior += w * Clamp(0.55*simMacro + 0.45*gProtos[i].counterTrendFailureRate,0.0,1.0);
      macroSum += w*simMacro;
      microSum += w*simMicro;
      wsum += w;
   }

   if(wsum>1e-8)
   {
      painRecurrence=Clamp(painRecurrence/wsum,0.0,1.0);
      macroTrapPrior=Clamp(macroTrapPrior/wsum,0.0,1.0);
      counterTrendPrior=Clamp(counterTrendPrior/wsum,0.0,1.0);
      regimeBreakPainPrior=Clamp(regimeBreakPainPrior/wsum,0.0,1.0);
      recoveryFalseStartRisk=Clamp(recoveryFalseStartRisk/wsum,0.0,1.0);
      deepBasketPainPrior=Clamp(deepBasketPainPrior/wsum,0.0,1.0);
      lateTrendFadePenalty=Clamp(lateTrendFadePenalty/wsum,0.0,1.0);
      painAgreement=Clamp(0.5*(macroSum/wsum) + 0.5*(microSum/wsum),0.0,1.0);
      macroMicroConflict=Clamp(MathAbs((macroSum/wsum) - (microSum/wsum)),0.0,1.0);
      painConfidence=Clamp(0.55*painAgreement + 0.45*Clamp(MathMin(wsum,3.0)/3.0,0.0,1.0),0.0,1.0);
   }
}

double ComputeOneRoundOpportunityScore(const string symbol,
                                       const int symIdx,
                                       const double &qBase[],
                                       const DecisionSupportContext &support)
{
   double qGap=Clamp(SafeDiv(ComputeDDQNActionGap(qBase,false),0.25,0.0),0.0,1.0);
   double spreadInv=1.0 - ComputeSpreadPressureScore(symbol,symIdx);
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double ddCalm=ComputeRecentDrawdownCalmBonus(symIdx);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double reversalConfirm=ComputeReversalConfirmationScore(symbol);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   int dirBest=ArgMaxDirectionalActionFromQ(qBase);
   double macroFit=0.5;
   if(dirBest==1)      macroFit=Clamp(0.5 + 0.5*macroBias, 0.0, 1.0);
   else if(dirBest==2) macroFit=Clamp(0.5 - 0.5*macroBias, 0.0, 1.0);

   double persistenceProb=Clamp(support.trendPersistenceProb,0.0,1.0);
   double reversalProb=Clamp(support.trendReversalProb,0.0,1.0);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double trendMode=Clamp(support.trendContinuationQuality,0.0,1.0);
   double reclaimMode=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double transitionMode=Clamp(support.reversalTransitionQuality,0.0,1.0);
   double modeDominance=Clamp(support.modeDominanceScore,0.0,1.0);
   double modeConflict=Clamp(support.modeConflictScore,0.0,1.0);
   double directionalBehaviorFit=Clamp(macroFit*persistenceProb + (1.0-macroFit)*reversalProb,0.0,1.0);

   double score=
      0.15*support.oneRoundPrior +
      0.10*support.supportConfidence +
      0.07*support.qMemoryAgreement +
      0.10*qGap +
      0.07*spreadInv +
      0.08*recentQuality +
      0.07*ddCalm +
      0.06*(1.0 - support.dangerProbability) +
      0.06*reversalConfirm +
      0.04*(1.0-trapRisk) +
      0.04*(1.0-regimeBreak) +
      0.04*macroFit +
      0.03*(1.0-macroMaturity) +
      0.02*(1.0-lateTrendTrap) +
      0.03*MathAbs(macroReclaim) +
      0.09*directionalBehaviorFit +
      0.05*(1.0-spikeRiskProb) +
      0.05*(1.0-expectedDepth) +
      0.06*trendMode +
      0.05*reclaimMode +
      0.03*modeDominance -
      0.06*transitionMode -
      0.03*modeConflict;

   return Clamp(score,0.0,1.0);
}

double ComputeDeepBasketRiskScore(const string symbol,
                                  const int symIdx,
                                  const DecisionSupportContext &support)
{
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double ddWorsen=Clamp(ComputeDrawdownWorseningScore(symIdx),0.0,1.0);
   double spread=ComputeSpreadPressureScore(symbol,symIdx);
   double replayRisk=Clamp(CurrentReplayRiskBias(symIdx),0.0,1.0);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double persistenceProb=Clamp(support.trendPersistenceProb,0.0,1.0);
   double reversalProb=Clamp(support.trendReversalProb,0.0,1.0);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double trendMode=Clamp(support.trendContinuationQuality,0.0,1.0);
   double reclaimMode=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double transitionMode=Clamp(support.reversalTransitionQuality,0.0,1.0);
   double modeConflict=Clamp(support.modeConflictScore,0.0,1.0);

   double score=
      0.16*support.addRiskPrior +
      0.14*support.dangerProbability +
      0.10*support.archiveCaution +
      0.09*replayRisk +
      0.08*recentDeepRate +
      0.06*ddWorsen +
      0.04*spread +
      0.03*(1.0 - recentQuality) +
      0.05*trapRisk +
      0.04*regimeBreak +
      0.03*macroContinuation +
      0.02*macroTransition +
      0.01*macroMaturity +
      0.01*MathAbs(macroReclaim) +
      0.07*spikeRiskProb +
      0.10*expectedDepth +
      0.04*MathMax(0.0,persistenceProb-reversalProb) +
      0.06*transitionMode +
      0.04*modeConflict -
      0.04*trendMode -
      0.03*reclaimMode;

   return Clamp(score,0.0,1.0);
}


void BuildAdaptiveEntryQualityDelta(const int symIdx,
                                    const int regime,
                                    const double &qBase[],
                                    const DecisionSupportContext &support,
                                    double &biasOut[])
{
   InitActionBias(biasOut);
   if(!UseQualityAdaptiveEntry) return;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(gPositionsCount[symIdx] > 0) return;

   string symbol=gSymbols[symIdx];
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double oneRoundScore=ComputeOneRoundOpportunityScore(symbol,symIdx,qBase,support);
   double deepRiskScore=ComputeDeepBasketRiskScore(symbol,symIdx,support);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double reversalConfirm=ComputeReversalConfirmationScore(symbol);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double qualityEdge=Clamp(oneRoundScore - deepRiskScore,-1.0,1.0);
   double periodicTilt=Clamp((recentQuality - 0.5)*2.0,-1.0,1.0);
   double persistenceProb=Clamp(support.trendPersistenceProb,0.0,1.0);
   double reversalProb=Clamp(support.trendReversalProb,0.0,1.0);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double trendMode=Clamp(support.trendContinuationQuality,0.0,1.0);
   double reclaimMode=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double transitionMode=Clamp(support.reversalTransitionQuality,0.0,1.0);
   double modeDominance=Clamp(support.modeDominanceScore,0.0,1.0);
   double modeConflict=Clamp(support.modeConflictScore,0.0,1.0);

   int dirBest=ArgMaxDirectionalActionFromQ(qBase);
   double dirMacroTrap=ComputeDirectionalMacroTrapRisk(dirBest, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   double trapPenalty=Clamp(0.28*trapRisk + 0.12*regimeBreak + 0.16*dirMacroTrap + 0.08*lateTrendTrap + 0.08*macroContinuation - 0.18*reversalConfirm + 0.08*spikeRiskProb + 0.12*expectedDepth, 0.0, 1.0);

   double macroAlignBonus=0.0;
   if(dirBest==1)      macroAlignBonus = MathMax(0.0, 0.65*macroBias + 0.18*macroReclaim);
   else if(dirBest==2) macroAlignBonus = MathMax(0.0, -0.65*macroBias - 0.18*macroReclaim);
   macroAlignBonus = Clamp(macroAlignBonus, 0.0, 1.0);

   double directionalBehaviorFit=Clamp(macroAlignBonus*persistenceProb + (1.0-macroAlignBonus)*reversalProb,0.0,1.0);

   double continuationTilt = Clamp(0.60*trendMode + 0.25*modeDominance + 0.15*macroAlignBonus, 0.0, 1.0);
   double reclaimTilt      = Clamp(0.65*reclaimMode + 0.20*MathAbs(macroReclaim) + 0.15*modeDominance, 0.0, 1.0);
   double transitionOffense= Clamp(0.45*reversalConfirm + 0.20*reversalProb + 0.15*MathAbs(macroReclaim) + 0.10*(1.0-trapRisk) + 0.10*(1.0-expectedDepth), 0.0, 1.0);
   double transitionCaution= Clamp(0.60*transitionMode + 0.20*modeConflict + 0.10*regimeBreak + 0.10*spikeRiskProb, 0.0, 1.0);

   double dirBoost=MathMax(0.0, 0.16*qualityEdge + 0.06*periodicTilt + 0.06*reversalConfirm + 0.06*macroAlignBonus + 0.07*directionalBehaviorFit - 0.10*trapPenalty);
   double holdBoost=MathMax(0.0, 0.18*(-qualityEdge) + 0.08*MathMax(0.0,-periodicTilt) + 0.08*deepRiskScore + 0.09*trapPenalty + 0.05*dirMacroTrap + 0.05*spikeRiskProb + 0.07*expectedDepth);
   double holdPenalty=MathMax(0.0, 0.10*qualityEdge + 0.04*MathMax(0.0,periodicTilt) + 0.04*macroAlignBonus + 0.04*directionalBehaviorFit);

   if(dirBest==1 || dirBest==2)
   {
      double sameDirBoost = dirBoost + 0.10*continuationTilt + 0.08*reclaimTilt + 0.04*modeDominance;
      double transitionDirBoost = dirBoost + 0.07*transitionMode*transitionOffense;
      double transitionHoldBoost = holdBoost + 0.14*transitionCaution + 0.10*transitionMode*(1.0-transitionOffense);

      if(dirBest==1)
      {
         if(macroBias >= 0.0)
            biasOut[1] += sameDirBoost;
         else
            biasOut[1] += transitionDirBoost;

         biasOut[2] -= 0.30*dirBoost + 0.08*continuationTilt;
         biasOut[0] += (macroBias >= 0.0 ? holdBoost : transitionHoldBoost) - holdPenalty;
      }
      else
      {
         if(macroBias <= 0.0)
            biasOut[2] += sameDirBoost;
         else
            biasOut[2] += transitionDirBoost;

         biasOut[1] -= 0.30*dirBoost + 0.08*continuationTilt;
         biasOut[0] += (macroBias <= 0.0 ? holdBoost : transitionHoldBoost) - holdPenalty;
      }
   }
   else
   {
      biasOut[0] += holdBoost + 0.08*transitionCaution - 0.03*modeDominance;
   }
}



double ComputeSpreadPressureScore(const string symbol,const int symIdx)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double spreadPts=MathMax(0.0,(ask-bid)/point);

   double atr=gATRslow_BaseVal[symIdx];
   if(atr<=1e-12) atr=gATRfast_BaseVal[symIdx];
   double atrPts=(atr>0.0 ? atr/point : 0.0);
   if(atrPts<=1e-6) return Clamp(spreadPts/10.0,0.0,1.0);

   return Clamp(12.0*SafeDiv(spreadPts,atrPts,0.0),0.0,1.0);
}

double ComputeSmartAddRiskScore(const string symbol,
                                const int symIdx,
                                const int regime,
                                const int basketDir,
                                const int positionsCount,
                                const double combinedTrend,
                                const double reversalRisk,
                                const bool extreme,
                                const DecisionSupportContext &support)
{
   double ddPct=GetSymbolFloatingDDPct(symIdx);
   double depth=Clamp(SafeDiv((double)MathMax(0,positionsCount-1),(double)MathMax(1,MaxTrades-1),0.0),0.0,1.0);
   double age=Clamp(SafeDiv(BasketAgeDays(symIdx),MathMax(AddGateAgeSoftDays,0.25),0.0),0.0,1.0);
   double danger=MathMax(Clamp(gPDanger[symIdx],0.0,1.0),support.dangerProbability);
   double addRisk=MathMax(support.addRiskPrior,CurrentReplayRiskBias(symIdx));
   double spread=ComputeSpreadPressureScore(symbol,symIdx);
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);

   double antiTrend=0.0;
   if(basketDir>0)
   {
      if(combinedTrend<0.0) antiTrend=1.0;
      else if(combinedTrend==0.0) antiTrend=0.25;
   }
   else if(basketDir<0)
   {
      if(combinedTrend>0.0) antiTrend=1.0;
      else if(combinedTrend==0.0) antiTrend=0.25;
   }

   double risk=0.20*ddPct +
               0.17*depth +
               0.11*age +
               0.18*danger +
               0.13*addRisk +
               0.10*antiTrend +
               0.04*Clamp(reversalRisk,0.0,1.0) +
               0.02*spread +
               0.03*(1.0-recentQuality) +
               0.02*recentDeepRate +
               0.04*trapRisk +
               0.02*regimeBreak +
               0.03*spikeRiskProb +
               0.04*expectedDepth;

   if(extreme) risk += 0.10;

   double sameBias=0.0;
   double oppBias=0.0;
   if(basketDir>0)
   {
      sameBias=support.actionDelta[1];
      oppBias =support.actionDelta[2];
   }
   else if(basketDir<0)
   {
      sameBias=support.actionDelta[2];
      oppBias =support.actionDelta[1];
   }

   if(support.actionDelta[0] > sameBias + 0.18) risk += 0.08;
   if(oppBias > sameBias + 0.20)                risk += 0.12;

   return Clamp(risk,0.0,1.5);
}

bool EvaluateSmartAddGate(const string symbol,
                          const int symIdx,
                          const int regime,
                          const double &state[],
                          const int basketDir,
                          const int positionsCount,
                          const double combinedTrend,
                          const double reversalRisk,
                          const bool extreme,
                          double &spacingMultOut,
                          double &lotScaleOut,
                          double &riskScoreOut)
{
   spacingMultOut=1.0;
   lotScaleOut=1.0;
   riskScoreOut=0.0;

   if(!UseSmartAddGate) return true;
   if(positionsCount<=0 || basketDir==0) return true;

   double qBase[];
   DQNForwardInference(symIdx,regime,state,qBase);

   DecisionSupportContext support;
   BuildDecisionSupportContext(symIdx,regime,state,qBase,false,basketDir,support);

   riskScoreOut=ComputeSmartAddRiskScore(symbol,
                                         symIdx,
                                         regime,
                                         basketDir,
                                         positionsCount,
                                         combinedTrend,
                                         reversalRisk,
                                         extreme,
                                         support);

   if(riskScoreOut >= AddGateRiskBlockThreshold)
      return false;

   if(riskScoreOut > AddGateRiskWidenThreshold)
   {
      double frac=Clamp(SafeDiv(riskScoreOut-AddGateRiskWidenThreshold,
                                MathMax(AddGateRiskBlockThreshold-AddGateRiskWidenThreshold,1e-6),0.0),0.0,1.0);
      spacingMultOut=1.0 + frac*(MathMax(AddGateSpacingMaxMult,1.0)-1.0);
      lotScaleOut   =1.0 - frac*(1.0-Clamp(AddGateLotMinScale,0.05,1.0));
   }

   return true;
}


bool GetLatestEpisodeForSymbol(const int symIdx, EpisodeMemory &epOut)
{
   int n=ArraySize(gEpisodeMemory);
   for(int i=n-1;i>=0;--i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      epOut=gEpisodeMemory[i];
      return true;
   }
   return false;
}

int DominantStrategyMode(const DecisionSupportContext &support)
{
   double tc=Clamp(support.trendContinuationQuality,0.0,1.0);
   double br=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double rt=Clamp(support.reversalTransitionQuality,0.0,1.0);
   if(tc<=1e-9 && br<=1e-9 && rt<=1e-9) return 0;
   if(tc>=br && tc>=rt) return 1;
   if(br>=tc && br>=rt) return 2;
   return 3;
}

double ComputeRegimeShiftAlert(const int symIdx,
                               const int regime,
                               const DecisionSupportContext &support,
                               const double macroBias)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   if(gSignalTrackBarTime[symIdx]<=0) return 0.0;

   double regimeShift=0.0;
   int prevReg=gSignalTrackRegime[symIdx];
   if(prevReg>=0)
   {
      int d=MathAbs(prevReg-regime);
      if(d>=2) regimeShift=1.0;
      else if(d==1) regimeShift=0.60;
   }

   int curMode=DominantStrategyMode(support);
   double modeShift=(gSignalTrackMode[symIdx]>0 && curMode>0 && gSignalTrackMode[symIdx]!=curMode ? 1.0 : 0.0);
   double transitionRise=MathMax(0.0, Clamp(support.reversalTransitionQuality,0.0,1.0) - Clamp(gSignalTrackTransition[symIdx],0.0,1.0));

   double prevMacro=gSignalTrackMacroBias[symIdx];
   double macroFlip=0.0;
   if(MathAbs(prevMacro)>=0.12 && MathAbs(macroBias)>=0.12 && ((prevMacro>0.0 && macroBias<0.0) || (prevMacro<0.0 && macroBias>0.0)))
      macroFlip=1.0;

   return Clamp(0.34*regimeShift +
                0.24*modeShift +
                0.20*transitionRise +
                0.12*macroFlip +
                0.10*Clamp(support.modeConflictScore,0.0,1.0),0.0,1.0);
}

int ExtractDirectionalCandidate(const double &qVals[])
{
   if(ArraySize(qVals)<3) return 0;
   int dir=(qVals[1] >= qVals[2] ? 1 : 2);
   double dirQ=qVals[dir];
   if(dirQ < qVals[0] - 0.16) return 0;
   return dir;
}

void UpdateDirectionalSignalTracking(const int symIdx,
                                     const int regime,
                                     const DecisionSupportContext &support,
                                     const double &qAdj[],
                                     const double macroBias)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   datetime barTime=support.barTime;
   int curMode=DominantStrategyMode(support);
   int candidate=ExtractDirectionalCandidate(qAdj);

   if(gSignalTrackBarTime[symIdx]==barTime)
   {
      if(candidate!=0)
         gSignalTrackDirCandidate[symIdx]=candidate;
      gSignalTrackRegime[symIdx]=regime;
      gSignalTrackMode[symIdx]=curMode;
      gSignalTrackMacroBias[symIdx]=macroBias;
      gSignalTrackTransition[symIdx]=Clamp(support.reversalTransitionQuality,0.0,1.0);
      return;
   }

   int persist=0;
   bool sameContext=(candidate!=0 &&
                     gSignalTrackDirCandidate[symIdx]==candidate &&
                     gSignalTrackMode[symIdx]==curMode &&
                     MathAbs(gSignalTrackRegime[symIdx]-regime)==0 &&
                     !((gSignalTrackMacroBias[symIdx]>0.12 && macroBias<-0.12) || (gSignalTrackMacroBias[symIdx]<-0.12 && macroBias>0.12)) &&
                     MathAbs(Clamp(gSignalTrackTransition[symIdx],0.0,1.0)-Clamp(support.reversalTransitionQuality,0.0,1.0))<0.24);

   if(candidate==0)
      persist=0;
   else if(sameContext)
      persist=MathMin(gSignalTrackPersistCount[symIdx]+1,8);
   else
      persist=1;

   gSignalTrackBarTime[symIdx]=barTime;
   gSignalTrackRegime[symIdx]=regime;
   gSignalTrackDirCandidate[symIdx]=candidate;
   gSignalTrackPersistCount[symIdx]=persist;
   gSignalTrackMode[symIdx]=curMode;
   gSignalTrackMacroBias[symIdx]=macroBias;
   gSignalTrackTransition[symIdx]=Clamp(support.reversalTransitionQuality,0.0,1.0);
}

double ComputeRecentWrongThesisPenalty(const string symbol,
                                      const int symIdx,
                                      const int action,
                                      const DecisionSupportContext &support,
                                      const double macroBias,
                                      const double macroReclaim,
                                      const double macroTransition)
{
   if(action<1 || action>2) return 0.0;

   EpisodeMemory ep;
   if(!GetLatestEpisodeForSymbol(symIdx,ep)) return 0.0;
   if(ep.endTime<=0) return 0.0;

   int recentSec=MathMax(PeriodSeconds(TF_EXEC)*18, 1800);
   if(recentSec<=0) recentSec=3600;
   if((TimeCurrent()-ep.endTime) > recentSec) return 0.0;

   double ugly=0.0;
   if(ep.openPositionsMax>=2) ugly += 0.30;
   ugly += 0.30*Clamp(SafeDiv(ep.maxDrawdownPct,MathMax(RewardV2CleanCycleMaxDD,1e-6),0.0),0.0,1.0);
   if(ep.inefficientRecovery>0) ugly += 0.18;
   if(ep.forcedStopLikeEvent>0) ugly += 0.12;
   if(ep.pnlFinal<=0.0) ugly += 0.10;
   ugly += 0.12*Clamp(1.0-ep.rewardEfficiency,0.0,1.0);
   ugly=Clamp(ugly,0.0,1.0);
   if(ugly<0.20) return 0.0;

   int dir=(action==1 ? 1 : -1);
   double sameRecent = (ep.basketDir!=0 && dir==ep.basketDir ? 1.0 : 0.0);
   double oppositeRecent = (ep.basketDir!=0 && dir!=ep.basketDir ? 1.0 : 0.0);
   double macroAgainst = MathMax(0.0,-dir*macroBias);
   double reclaimAgainst = MathMax(0.0,-dir*macroReclaim);
   double transitionAmb = Clamp(0.60*macroTransition + 0.40*support.modeConflictScore,0.0,1.0);

   return Clamp(ugly*(0.35*sameRecent +
                      0.20*oppositeRecent +
                      0.20*macroAgainst +
                      0.10*reclaimAgainst +
                      0.10*transitionAmb +
                      0.05*Clamp(support.expectedBasketDepth,0.0,1.0)),0.0,1.0);
}

int ComputeDirectionalRegimeState(const string symbol,
                                  const int symIdx,
                                  const int regime,
                                  const DecisionSupportContext &support,
                                  double &macroBiasOut,
                                  double &macroReclaimOut,
                                  double &regimeShiftOut,
                                  double &clarityOut,
                                  bool &neutralSafeOut)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   macroBiasOut=macroBias;
   macroReclaimOut=macroReclaim;

   regimeShiftOut=ComputeRegimeShiftAlert(symIdx,regime,support,macroBias);

   double breakoutSigned = Clamp((macroBias + 0.65*macroReclaim),-1.0,1.0) * Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double directionEvidence = Clamp(0.34*macroBias +
                                    0.16*macroReclaim +
                                    0.16*(Clamp(support.trendPersistenceProb,0.0,1.0)-Clamp(support.trendReversalProb,0.0,1.0)) +
                                    0.14*(Clamp(support.trendContinuationQuality,0.0,1.0)-Clamp(support.reversalTransitionQuality,0.0,1.0)) +
                                    0.12*breakoutSigned +
                                    0.08*Clamp(support.modeDominanceScore*(Clamp(support.trendContinuationQuality,0.0,1.0)-Clamp(support.reversalTransitionQuality,0.0,1.0)),-1.0,1.0), -1.0, 1.0);

   clarityOut = Clamp(MathAbs(directionEvidence) *
                      (1.0-0.55*Clamp(support.modeConflictScore,0.0,1.0)) *
                      (1.0-0.45*Clamp(regimeShiftOut,0.0,1.0)) *
                      (1.0-0.35*Clamp(support.reversalTransitionQuality,0.0,1.0)), 0.0, 1.0);

   double neutralStableScore = Clamp(0.24*(1.0-Clamp(support.spikeRiskProb,0.0,1.0)) +
                                     0.20*(1.0-Clamp(support.expectedBasketDepth,0.0,1.0)) +
                                     0.16*Clamp(support.oneRoundPrior,0.0,1.0) +
                                     0.12*Clamp(support.supportConfidence,0.0,1.0) +
                                     0.12*(1.0-Clamp(support.modeConflictScore,0.0,1.0)) +
                                     0.08*(1.0-Clamp(support.painRecurrenceRisk,0.0,1.0)) +
                                     0.08*(1.0-Clamp(regimeShiftOut,0.0,1.0)), 0.0, 1.0);
   neutralSafeOut = (neutralStableScore >= 0.52 &&
                     support.reversalTransitionQuality < 0.70 &&
                     support.spikeRiskProb < 0.72 &&
                     support.expectedBasketDepth < 0.72);

   if(clarityOut >= 0.22 && directionEvidence > 0.14)
      return DIR_STATE_UP;
   if(clarityOut >= 0.22 && directionEvidence < -0.14)
      return DIR_STATE_DOWN;
   return DIR_STATE_NEUTRAL;
}

double ComputeFlexibleDirectionalDisciplinePenalty(const string symbol,
                                                   const int symIdx,
                                                   const int regime,
                                                   const int action,
                                                   const DecisionSupportContext &support,
                                                   const bool forcedEntry)
{
   if(action<1 || action>2) return 0.0;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double reversalConfirm=ComputeReversalConfirmationScore(symbol);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double macroTrap=ComputeDirectionalMacroTrapRisk(action, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   double wrongThesis=ComputeRecentWrongThesisPenalty(symbol,symIdx,action,support,macroBias,macroReclaim,macroTransition);

   double macroBiasDir=0.0,macroReclaimDir=0.0,regimeShiftAlert=0.0,clarity=0.0;
   bool neutralSafe=false;
   int dirState=ComputeDirectionalRegimeState(symbol,symIdx,regime,support,macroBiasDir,macroReclaimDir,regimeShiftAlert,clarity,neutralSafe);

   int dir=(action==1 ? 1 : -1);
   double alignBias=dir*macroBias;
   double alignReclaim=dir*macroReclaim;
   double continuationDominant=(support.trendContinuationQuality > support.breakoutReclaimQuality + 0.04 &&
                                support.trendContinuationQuality > support.reversalTransitionQuality + 0.04 ? 1.0 : 0.0);
   double reclaimDominant=(support.breakoutReclaimQuality > support.trendContinuationQuality + 0.04 &&
                           support.breakoutReclaimQuality > support.reversalTransitionQuality + 0.04 ? 1.0 : 0.0);

   double penalty=0.0;

   if(dirState==DIR_STATE_UP || dirState==DIR_STATE_DOWN)
   {
      bool againstClear=((dirState==DIR_STATE_UP && action==2) || (dirState==DIR_STATE_DOWN && action==1));
      if(againstClear)
      {
         double counterConfirm = Clamp(0.42*reversalConfirm +
                                       0.20*Clamp(support.trendReversalProb,0.0,1.0) +
                                       0.14*MathMax(0.0,alignReclaim) +
                                       0.12*(1.0- Clamp(support.expectedBasketDepth,0.0,1.0)) +
                                       0.12*Clamp(support.reversalTransitionQuality,0.0,1.0),0.0,1.0);
         penalty += 0.58 + 0.28*MathMax(0.0,0.62-counterConfirm) + 0.18*macroTrap + 0.12*regimeShiftAlert;
      }
      else
      {
         penalty += 0.10*MathMax(0.0,-alignBias);
         penalty += 0.08*MathMax(0.0,-alignReclaim);
         penalty += 0.08*regimeShiftAlert;
         if(reclaimDominant>0.5 && alignReclaim< -0.05) penalty += 0.14;
         if(continuationDominant>0.5 && alignBias< -0.05) penalty += 0.12;
      }
   }
   else
   {
      if(!neutralSafe)
      {
         penalty += 0.20 + 0.14*Clamp(support.modeConflictScore,0.0,1.0) +
                    0.12*Clamp(support.spikeRiskProb,0.0,1.0) +
                    0.12*Clamp(support.expectedBasketDepth,0.0,1.0) +
                    0.10*regimeShiftAlert;
      }
      else
      {
         penalty += 0.10*regimeBreak + 0.10*trapRisk + 0.08*wrongThesis;
         penalty += 0.12*MathMax(0.0,-alignBias-0.10);
      }
   }

   penalty += 0.16*wrongThesis;
   penalty += 0.10*trapRisk;
   penalty += 0.08*macroTrap;

   if(forcedEntry)
      penalty *= 0.78;

   return Clamp(penalty,0.0,1.25);
}

void ApplyFlexibleDirectionalDiscipline(const string symbol,
                                        const int symIdx,
                                        const int regime,
                                        const DecisionSupportContext &support,
                                        const bool forcedEntry,
                                        double &qAdj[])
{
   if(ArraySize(qAdj)<3) return;

   for(int action=1; action<=2 && action<ArraySize(qAdj); ++action)
   {
      double penalty=ComputeFlexibleDirectionalDisciplinePenalty(symbol,symIdx,regime,action,support,forcedEntry);
      if(penalty<=0.0) continue;
      qAdj[action] -= penalty;
      if(!forcedEntry)
         qAdj[0] += 0.22*MathMin(penalty,1.10);
   }
}

void ApplyConfirmedSetupAccelerator(const string symbol,
                                    const int symIdx,
                                    const int regime,
                                    const DecisionSupportContext &support,
                                    double &qAdj[])
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(gPositionsCount[symIdx]>0) return;
   if(ArraySize(qAdj)<3) return;

   double macroBias=0.0,macroReclaim=0.0,regimeShiftAlert=0.0,clarity=0.0;
   bool neutralSafe=false;
   int dirState=ComputeDirectionalRegimeState(symbol,symIdx,regime,support,macroBias,macroReclaim,regimeShiftAlert,clarity,neutralSafe);

   UpdateDirectionalSignalTracking(symIdx,regime,support,qAdj,macroBias);

   int dir=ExtractDirectionalCandidate(qAdj);
   if(dir<1 || dir>2) return;
   if(gSignalTrackPersistCount[symIdx] < 2) return;

   bool modeOkay=(DominantStrategyMode(support)==1 || DominantStrategyMode(support)==2 || (dirState==DIR_STATE_NEUTRAL && neutralSafe));
   if(!modeOkay) return;

   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double oneRound=Clamp(support.oneRoundPrior,0.0,1.0);
   double supportConf=Clamp(support.supportConfidence,0.0,1.0);
   double qDir=qAdj[dir];
   double qHold=qAdj[0];
   double dirEdge=Clamp((qDir-qHold+0.10)/0.28,0.0,1.0);

   double alignScore=0.5;
   if(dirState==DIR_STATE_UP)
      alignScore=(dir==1 ? Clamp(0.65 + 0.30*clarity + 0.10*MathMax(0.0,macroReclaim),0.0,1.0) : 0.0);
   else if(dirState==DIR_STATE_DOWN)
      alignScore=(dir==2 ? Clamp(0.65 + 0.30*clarity + 0.10*MathMax(0.0,-macroReclaim),0.0,1.0) : 0.0);
   else
      alignScore=(neutralSafe ? 0.58 : 0.0);

   double setupStrength=Clamp(0.20*Clamp(support.trendContinuationQuality,0.0,1.0) +
                              0.16*Clamp(support.breakoutReclaimQuality,0.0,1.0) +
                              0.16*Clamp(support.modeDominanceScore,0.0,1.0) +
                              0.12*alignScore +
                              0.12*oneRound +
                              0.10*supportConf +
                              0.08*dirEdge +
                              0.06*Clamp((double)gSignalTrackPersistCount[symIdx]/3.0,0.0,1.0),0.0,1.0);

   double caution=Clamp(0.24*Clamp(support.reversalTransitionQuality,0.0,1.0) +
                        0.18*Clamp(support.modeConflictScore,0.0,1.0) +
                        0.16*Clamp(support.spikeRiskProb,0.0,1.0) +
                        0.14*expectedDepth +
                        0.14*trapRisk +
                        0.14*Clamp(regimeShiftAlert,0.0,1.0),0.0,1.0);

   if(setupStrength < 0.54 || caution > 0.56) return;

   double boost=Clamp(0.08 + 0.18*(setupStrength-caution),0.0,0.20);
   qAdj[dir] += boost;
   qAdj[0]   -= 0.35*boost;
}

void ComputeLiveTrendBiasAndReversal(const string symbol,
                                     const int symIdx,
                                     const int regime,
                                     const DecisionSupportContext &support,
                                     double &combinedTrend,
                                     double &reversalRisk)
{
   double macroBias=0.0,macroReclaim=0.0,regimeShiftAlert=0.0,clarity=0.0;
   bool neutralSafe=false;
   int dirState=ComputeDirectionalRegimeState(symbol,symIdx,regime,support,macroBias,macroReclaim,regimeShiftAlert,clarity,neutralSafe);

   double stateBias=(dirState==DIR_STATE_UP ? 1.0 : (dirState==DIR_STATE_DOWN ? -1.0 : 0.0));
   combinedTrend = Clamp(0.42*macroBias +
                         0.16*macroReclaim +
                         0.14*(Clamp(support.trendPersistenceProb,0.0,1.0)-Clamp(support.trendReversalProb,0.0,1.0)) +
                         0.10*(Clamp(support.trendContinuationQuality,0.0,1.0)-Clamp(support.breakoutReclaimQuality,0.0,1.0)) +
                         0.10*stateBias*clarity -
                         0.10*Clamp(support.modeConflictScore,0.0,1.0), -1.0, 1.0);

   reversalRisk = Clamp(0.26*Clamp(support.reversalTransitionQuality,0.0,1.0) +
                        0.16*Clamp(support.modeConflictScore,0.0,1.0) +
                        0.14*Clamp(support.trendReversalProb,0.0,1.0) +
                        0.12*Clamp(support.spikeRiskProb,0.0,1.0) +
                        0.12*Clamp(support.macroMicroConflict,0.0,1.0) +
                        0.10*Clamp(support.painRecurrenceRisk,0.0,1.0) +
                        0.10*Clamp(regimeShiftAlert,0.0,1.0), 0.0, 1.0);
}


int DQNSelectForcedEntryAction(const int symIdx,const int regime,double &state[])
{
   double qBase[];
   DQNForwardInference(symIdx,regime,state,qBase);

   double qAdj[];
   ArrayResize(qAdj, ArraySize(qBase));
   for(int a=0;a<ArraySize(qBase);a++) qAdj[a]=qBase[a];

   double totalDelta[];
   InitActionBias(totalDelta);

   DecisionSupportContext support;
   BuildDecisionSupportContext(symIdx, regime, state, qBase, true, 0, support);
   ApplyDecisionSupportDelta(support,totalDelta);

   ApplyBudgetedSubordinateBias(symIdx, true, qBase, totalDelta, qAdj);

   string symbol=gSymbols[symIdx];
   ApplyFlexibleDirectionalDiscipline(symbol, symIdx, regime, support, true, qAdj);

   if(ActionCount >= 3)
      return (qAdj[1] >= qAdj[2] ? 1 : 2);

   if(ActionCount >= 2)
      return 1;

   return 0;
}


bool CheckProfitPauseTrigger()
{
   if(!UseProfitPause) return false;
   if(ProfitPauseTargetAmount <= 0.0) return false;
   return (gProfitCycleClosedProfit >= ProfitPauseTargetAmount);
}

void StartProfitPause()
{
   int sec = MathMax(0, ProfitPauseDurationHours) * 3600;
   if(sec <= 0)
      gProfitPauseResumeTime = 0;
   else
      gProfitPauseResumeTime = TimeCurrent() + sec;
}

double GetSymbolFloatingLossMoney(const string symbol,const int magic)
{
   double pnl = CalculatePositionsPnL(symbol,magic);
   if(pnl >= 0.0) return 0.0;
   return -pnl;
}

bool HandleEquityLossStop()
{
   if(!CheckEquityLossStop())
      return false;

   if(EquityLossStopClosePositions)
   {
      CloseAllPositions();
      CountOpenPositions();
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      gFirstTradeTime[i]=0;
      gLastTradeOpenTime[i]=TimeCurrent();
      ResetForcedEntryState(i);
   }
   // reset peak reference so old DD does not keep blocking
   maxEquity = GetEAEquity();

   // keep reward baseline aligned too
   gTickEquityBaseline = maxEquity;
   gRewardBaselineTick = 0;

   return true;
}

void CountOpenPositions()
{
   for(int i=0;i<gSymbolCount;i++)
   {
      gPositionsCount[i]=0;
      gTrades[i].Clear();
      gTradeLots[i].Clear();

      gCachedSymbolPnL[i]=0.0;
      gHasPositionTypeCache[i]=false;
      gBasketDirCache[i]=0;
      gBasketAvgPriceCache[i]=0.0;
      gBasketAvgValid[i]=false;
   }

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym=PositionGetString(POSITION_SYMBOL);
      long mg=PositionGetInteger(POSITION_MAGIC);
      double price=PositionGetDouble(POSITION_PRICE_OPEN);
      double vol=PositionGetDouble(POSITION_VOLUME);
      double pnl=PositionGetDouble(POSITION_PROFIT);

      int idx=SymbolIndex(sym);
      if(idx<0) continue;
      if((int)mg!=gMagics[idx]) continue;

      gPositionsCount[idx]++;
      gTrades[idx].Add(price);
      gTradeLots[idx].Add(vol);
      gCachedSymbolPnL[idx] += pnl;

      if(!gHasPositionTypeCache[idx])
      {
         ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         gHasPositionTypeCache[idx]=true;
         gBasketDirCache[idx]=(pt==POSITION_TYPE_BUY ? +1 : -1);
      }
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      if(gTrades[i].Total()>0 && gTradeLots[i].Total()==gTrades[i].Total())
      {
         double weighted=0.0;
         double totalLots=0.0;

         for(int j=0;j<gTrades[i].Total();j++)
         {
            double vol=gTradeLots[i].At(j);
            weighted += gTrades[i].At(j)*vol;
            totalLots += vol;
         }

         if(totalLots>0.0)
         {
            gBasketAvgPriceCache[i]=weighted/totalLots;
            gBasketAvgValid[i]=true;
         }
      }
   }
}

double BasketAvgPrice(CArrayDouble &trades,CArrayDouble &tradeLots,double fallback)
{
   if(trades.Total()<=0 || tradeLots.Total()!=trades.Total()) return fallback;

   double weighted=0.0;
   double totalLots=0.0;
   for(int i=0;i<trades.Total();i++)
   {
      double vol=tradeLots.At(i);
      weighted += trades.At(i)*vol;
      totalLots += vol;
   }

   if(totalLots<=0.0) return fallback;
   return weighted / totalLots;
}

double GetEquityBudgetScale()
{
   if(EquityBudget<=0.0) return 1.0;
   double bal=AccountInfoDouble(ACCOUNT_BALANCE);
   if(bal<=1e-9) return 1.0;
   double scale=EquityBudget/bal;
   if(scale>1.0) scale=1.0;
   if(scale<0.01) scale=0.01;
   return scale;
}

double GetAtrRatioCached_Base(const string symbol)
{
   int idx=SymbolIndex(symbol);
   if(idx<0) return 1.0;

   datetime bt=iTime(symbol,BaseTF,0);
   if(bt!=gATRLastBarTime[idx])
   {
      double f=0.0,s=0.0;
      if(Copy1(hATRfast_Base[idx],0,f)) gATRfast_BaseVal[idx]=f;
      if(Copy1(hATRslow_Base[idx],0,s)) gATRslow_BaseVal[idx]=s;

      double ratio=1.0;
      if(s>0.0) ratio=Clamp(f/s,0.1,5.0);

      gATRratioCache[idx]=ratio;
      gATRLastBarTime[idx]=bt;
   }
   return gATRratioCache[idx];
}

int RegimeIndexFromRatio(const double atrRatio)
{
   if(!UseRegimeBank) return 0;
   if(atrRatio < RegimeLowThresh)  return 0;
   if(atrRatio > RegimeHighThresh) return 2;
   return 1;
}

bool ShouldTrainAllRegimesNow()
{
   if(!TrainAllRegimes) return false;
   if(!TrainAllRegimesWarmupOnly) return true;
   return (ArraySize(gReplay) < MathMax(100, RegimeWarmupReplayCount));
}

double RegimeContextSimilarityBonus(const double currentAtrRatio,const double protoAtrRatio)
{
   if(!UseRegimeBank) return 0.0;
   int cur = RegimeIndexFromRatio(currentAtrRatio);
   int prv = RegimeIndexFromRatio(protoAtrRatio);
   if(cur==prv) return DangerRegimeMatchBonus;
   if(MathAbs(cur-prv)>=2) return -DangerRegimeMismatchPenalty;
   return 0.0;
}

void ComputeSoftRegimeInferenceWeights(const double atrRatio,double &wOut[])
{
   ArrayResize(wOut, REGIME_COUNT);
   for(int r=0;r<REGIME_COUNT;r++) wOut[r]=0.0;
   if(!UseRegimeBank)
   {
      wOut[0]=1.0;
      return;
   }

   double span = MathMax(0.05, RegimeHighThresh - RegimeLowThresh);
   double width = MathMax(0.05, RegimeInferenceBlendWidth * MathMax(1.0, span));
   double centers[REGIME_COUNT];
   centers[0] = RegimeLowThresh - 0.50 * span;
   centers[1] = 0.50 * (RegimeLowThresh + RegimeHighThresh);
   centers[2] = RegimeHighThresh + 0.50 * span;

   double sumW=0.0;
   for(int r=0;r<REGIME_COUNT;r++)
   {
      double d = (atrRatio - centers[r]) / width;
      wOut[r] = MathExp(-0.5 * d * d);
      sumW += wOut[r];
   }
   if(sumW <= 1e-12)
   {
      int hard = RegimeIndexFromRatio(atrRatio);
      for(int r=0;r<REGIME_COUNT;r++) wOut[r] = (r==hard ? 1.0 : 0.0);
      return;
   }
   for(int r=0;r<REGIME_COUNT;r++) wOut[r] /= sumW;
}

void DQNForwardInference(int symIdx,int regime,const double &state[],double &qOut[])
{
   if(!UseRegimeBank || !UseSoftRegimeInference)
   {
      DQNForward(symIdx, regime, state, qOut);
      return;
   }

   double atrRatio = GetAtrRatioCached_Base(gSymbols[symIdx]);
   double rw[];
   ComputeSoftRegimeInferenceWeights(atrRatio, rw);

   ArrayResize(qOut, ActionCount);
   for(int a=0;a<ActionCount;a++) qOut[a]=0.0;

   for(int r=0;r<REGIME_COUNT;r++)
   {
      if(r>=ArraySize(rw) || rw[r] <= 1e-12) continue;
      double qTmp[];
      DQNForward(symIdx, r, state, qTmp);
      if(ArraySize(qTmp) != ActionCount) continue;
      for(int a=0;a<ActionCount;a++) qOut[a] += rw[r] * qTmp[a];
   }
}


double SmoothRange(const double prev,const double cur)
{
   double a=Clamp(GridRangeSmooth,0.0,1.0);
   if(a<=0.0) return cur;
   if(prev<=0.0) return cur;
   return (1.0-a)*prev + a*cur;
}

double GetFallbackGridStep(const string symbol)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double step=(double)DefaultPips*point;
   if(step<=0.0) step=point*10.0;
   return step;
}


double ChannelWidth_BaseTF(const string symbol,const int barsBack,const int shift=1)
{
   int lookback=MathMax(2,barsBack);
   int startShift=MathMax(1,shift);
   int bars=iBars(symbol,BaseTF);
   if(bars<=startShift+lookback) return 0.0;

   double hi=-DBL_MAX, lo=DBL_MAX;
   for(int i=startShift; i<startShift+lookback; i++)
   {
      double h=iHigh(symbol,BaseTF,i);
      double l=iLow(symbol,BaseTF,i);
      if(h>hi) hi=h;
      if(l<lo) lo=l;
   }

   if(hi<=lo) return 0.0;
   return (hi-lo);
}
void ComputeRollingChannelWidthStats(const string symbol,
                                     const int widthLookback,
                                     const int statsBars,
                                     double &currentWidth,
                                     double &meanWidth,
                                     double &sigmaWidth)
{
   currentWidth=0.0;
   meanWidth=0.0;
   sigmaWidth=0.0;

   int lookback=MathMax(2,widthLookback);
   int samples=MathMax(5,statsBars);
   int bars=iBars(symbol,BaseTF);
   if(bars<=lookback+samples+2)
      return;

   double sum=0.0, sumSq=0.0;
   int used=0;
   for(int shift=1; shift<=samples; shift++)
   {
      double w=ChannelWidth_BaseTF(symbol,lookback,shift);
      if(w<=0.0) continue;

      if(used==0)
         currentWidth=w;

      sum   += w;
      sumSq += w*w;
      used++;
   }

   if(used<=0)
      return;

   meanWidth=sum/(double)used;
   if(currentWidth<=0.0)
      currentWidth=meanWidth;

   if(used<2)
      return;

   double var=(sumSq/(double)used) - meanWidth*meanWidth;
   if(var>0.0)
      sigmaWidth=MathSqrt(var);
}

double ComputeAdaptiveBaseGridStep(const string symbol,const double factor)
{
   if(factor<=0.0) return 0.0;

   double currentWidth=0.0, meanWidth=0.0, sigmaWidth=0.0;
   ComputeRollingChannelWidthStats(symbol,Depth,GridWidthStatsBars,currentWidth,meanWidth,sigmaWidth);

   double baseWidth=currentWidth;
   if(baseWidth<=0.0)
      baseWidth=ChannelWidth_BaseTF(symbol,Depth,1);

   if(baseWidth<=0.0)
      return 0.0;

   if(meanWidth>0.0 && currentWidth>0.0 && currentWidth<meanWidth)
   {
      double deficit=(meanWidth-currentWidth)/MathMax(meanWidth,1e-10);
      if(deficit<0.0) deficit=0.0;

      double weightRoll=deficit;
      if(weightRoll>GridRollingMeanMaxWeight)
         weightRoll=GridRollingMeanMaxWeight;

      baseWidth=currentWidth*(1.0-weightRoll) + meanWidth*weightRoll;
   }

   return baseWidth/factor;
}

double ComputeAdaptiveGridStep(const string symbol,
                               const int symIdx,
                               const int positionsCount,
                               const double baseStep)
{
   if(baseStep<=0.0) return baseStep;
   if(!UseStdevGridExpansion) return baseStep;

   int nextPos=((positionsCount>0) ? (positionsCount+1) : 1);
   int startPos=MathMax(4,GridStdevStartPosition);
   if(nextPos<startPos) return baseStep;

   int maxPos=MathMax(startPos,GridStdevMaxPosition);
   if(nextPos>maxPos) nextPos=maxPos;

   int depthStep=nextPos-startPos+1;
   if(depthStep<=0) return baseStep;

   double sigmaStep=gGridWidthSigmaCache[symIdx];
   if(sigmaStep<=0.0) return baseStep;

   double sigmaMult=GridStdevCoeff*(double)depthStep;
   if(sigmaMult<0.0) sigmaMult=0.0;
   if(sigmaMult>GridStdevMaxSigmaMult) sigmaMult=GridStdevMaxSigmaMult;

   double adaptive=baseStep + sigmaStep*sigmaMult;
   if(adaptive<=0.0) adaptive=baseStep;

   return adaptive;
}


void UpdateMultiChannelGridStats(const int symIdx)
{
   string sym=gSymbols[symIdx];
   datetime baseBar=iTime(sym,BaseTF,0);
   if(baseBar==0) return;

   if(baseBar!=gGridLastBar[symIdx])
   {
      double step=0.0;
      double sigmaStep=0.0;

      if(UseDynamicPips)
      {
         step=ComputeAdaptiveBaseGridStep(sym,PipsFactor);

         double curWidth=0.0, meanWidth=0.0, sigmaWidth=0.0;
         ComputeRollingChannelWidthStats(sym,Depth,GridWidthStatsBars,curWidth,meanWidth,sigmaWidth);
         if(PipsFactor>0.0 && sigmaWidth>0.0)
            sigmaStep=sigmaWidth/PipsFactor;
      }

      if(step<=0.0)
         step=GetFallbackGridStep(sym);

      double smoothed=SmoothRange(gGridStepCache[symIdx],step);
      if(smoothed<=0.0) smoothed=step;

      gGridStepCache[symIdx]=smoothed;
      gGridWidthSigmaCache[symIdx]=sigmaStep;
      gGridActiveStep[symIdx]=smoothed;
      gGridActiveChannel[symIdx]=0;   // single-step / BaseTF mode
      gGridLastBar[symIdx]=baseBar;
   }
}


double GetGridStepCached(const int symIdx)
{
   UpdateMultiChannelGridStats(symIdx);

   double step=gGridStepCache[symIdx];
   if(step<=0.0)
      step=GetFallbackGridStep(gSymbols[symIdx]);

   gGridActiveStep[symIdx]=step;
   gGridActiveChannel[symIdx]=0;
   return step;
}

double GetCloseChannelStep(const int symIdx,const int positionsCount)
{
   double step=GetGridStepCached(symIdx);
   gGridActiveStep[symIdx]=step;
   gGridActiveChannel[symIdx]=0;
   return step;
}


bool IsPersistentExtremeDD(const int symIdx)
{
   if(ExtremeDDMoneyTrigger<=0.0 || ExtremeDDHoursTrigger<=0)
      return false;

   double eq=GetEAEquity();
   double ddMoney=(maxEquity>eq ? (maxEquity-eq) : 0.0);

   if(ddMoney>=ExtremeDDMoneyTrigger)
   {
      if(!gExtremeDDArmed[symIdx])
      {
         gExtremeDDArmed[symIdx]=true;
         gExtremeDDStart[symIdx]=TimeCurrent();
      }

      int secNeed=ExtremeDDHoursTrigger*3600;
      if(gExtremeDDStart[symIdx]>0 && (TimeCurrent()-gExtremeDDStart[symIdx])>=secNeed)
         return true;
   }
   else
   {
      gExtremeDDArmed[symIdx]=false;
      gExtremeDDStart[symIdx]=0;
   }

   return false;
}

void UpdateGridStatusPanel(const int symIdx)
{
   if(!ShowGridStatusPanel) return;

   string sym=gSymbols[symIdx];
   double point=SymbolInfoDouble(sym,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double stepPts=GetGridStepCached(symIdx)/point;
   double closePts=GetCloseChannelStep(symIdx,MathMax(1,gPositionsCount[symIdx]))/point;

   string txt;
   txt =
      "Leg count: " + IntegerToString(gPositionsCount[symIdx]) + "\n" +
      "Grid mode: SINGLE-STEP\n" +
      "Grid TF: " + EnumToString(BaseTF) + "\n" +
      "Add step: " + DoubleToString(stepPts,1) + " pts\n" +
      "Close ref: " + DoubleToString(closePts,1) + " pts";

   if(ProfitPauseActive())
      txt += "\nPAUSED until: " + TimeToString(gProfitPauseResumeTime, TIME_DATE|TIME_MINUTES);
   else if(UseProfitPause)
      txt += "\nProfit cycle: " + DoubleToString(gProfitCycleClosedProfit,2) +
             " / " + DoubleToString(ProfitPauseTargetAmount,2);

   if(UseReplayDiagnostics && ReplayDiagnosticsInStatusPanel)
      txt += "\n" + BuildReplayDiagnosticsText(symIdx);

   if(ObjectFind(0,gGridPanelName)<0)
   {
      ObjectCreate(0,gGridPanelName,OBJ_LABEL,0,0,0);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_CORNER,GridPanelCorner);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_XDISTANCE,GridPanelX);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_YDISTANCE,GridPanelY);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_FONTSIZE,10);
      ObjectSetString(0,gGridPanelName,OBJPROP_FONT,"Arial");
      ObjectSetInteger(0,gGridPanelName,OBJPROP_COLOR,clrWhite);
   }

   ObjectSetString(0,gGridPanelName,OBJPROP_TEXT,txt);
}


void GetZoneFeatures(string symbol,double lastClose,
                     double &zoneTypeNorm,double &zoneLocNorm,double &zoneStatusNorm)
{
   zoneTypeNorm   = 0.5;
   zoneLocNorm    = 0.5;
   zoneStatusNorm = 0.5;

   int total = ArraySize(zones);
   if(total<=0) return;

   int bestIdx=-1;
   double bestDist=DBL_MAX;
   datetime now=TimeCurrent();

   for(int i=0;i<total;i++)
   {
      if(zones[i].symbol!=symbol) continue;
      if(now>zones[i].endTime) continue;

      double centre=0.5*(zones[i].high+zones[i].low);
      double dist=MathAbs(lastClose-centre);
      if(dist<bestDist){bestDist=dist; bestIdx=i;}
   }
   if(bestIdx<0) return;

   SDZone z=zones[bestIdx];
   double h=z.high, l=z.low;
   double height=h-l;
   if(height<=0.0) return;

   zoneTypeNorm = (z.isDemand ? 0.0 : 1.0);

   double loc = (lastClose - l)/height;
   zoneLocNorm = Clamp(loc,0.0,1.0);

   if(z.broken) zoneStatusNorm=1.0;
   else if(z.tested) zoneStatusNorm=0.5;
   else zoneStatusNorm=0.0;
}

void ComputeD1TrendFeaturesRaw(string symbol,double &d1DistNorm,double &d1SlopeNorm,double &d1SideDurNorm)
{
   d1DistNorm=0.0; d1SlopeNorm=0.0; d1SideDurNorm=0.0;

   int barsD1=iBars(symbol,PERIOD_D1);
   int maxNeed=MathMax(D1_SlopeLookbackDays,D1_SideLookbackDays)+1;
   if(barsD1<=maxNeed || D1_EMA_Period<=1) return;

   int emaHandle=iMA(symbol,PERIOD_D1,D1_EMA_Period,0,MODE_EMA,PRICE_CLOSE);
   if(emaHandle==INVALID_HANDLE) return;

   double emaBuf[];
   ArraySetAsSeries(emaBuf,true);
   int copied=CopyBuffer(emaHandle,0,0,maxNeed,emaBuf);
   IndicatorRelease(emaHandle);
   if(copied<maxNeed) return;

   double emaCurr=emaBuf[0];

   double atrD1=0.0;
   int atrHandle=iATR(symbol,PERIOD_D1,ATR_SlowPeriod);
   if(atrHandle!=INVALID_HANDLE)
   {
      double buf[1];
      if(CopyBuffer(atrHandle,0,0,1,buf)>0) atrD1=buf[0];
      IndicatorRelease(atrHandle);
   }

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   if(atrD1<=0.0) atrD1=point*10000.0;

   double closeCurr=iClose(symbol,PERIOD_D1,0);
   d1DistNorm=Clamp((closeCurr-emaCurr)/atrD1,-5.0,5.0);

   int lookback=MathMin(D1_SlopeLookbackDays,barsD1-1);
   if(lookback>0 && lookback<copied)
   {
      double emaOld=emaBuf[lookback];
      d1SlopeNorm=Clamp((emaCurr-emaOld)/(atrD1*lookback),-2.0,2.0);
   }

   int maxSide=MathMin(D1_SideLookbackDays,barsD1-1);
   int sideCurr=(closeCurr>=emaCurr?1:-1);
   int count=0;
   for(int i=0;i<maxSide;i++)
   {
      double c=iClose(symbol,PERIOD_D1,i);
      double e=(i<copied?emaBuf[i]:emaCurr);
      int side=(c>=e?1:-1);
      if(side==sideCurr) count++;
      else break;
   }

   if(D1_MaxSideDurationDays>0.0)
      d1SideDurNorm=Clamp((double)count/D1_MaxSideDurationDays,0.0,1.0);
}

void GetD1TrendFeatures(string symbol,double &d1DistNorm,double &d1SlopeNorm,double &d1SideDurNorm)
{
   int idx=SymbolIndex(symbol);
   if(idx<0){ d1DistNorm=0.0; d1SlopeNorm=0.0; d1SideDurNorm=0.0; return; }

   datetime bt=iTime(symbol,PERIOD_D1,0);
   if(bt!=gD1LastBarTime[idx])
   {
      ComputeD1TrendFeaturesRaw(symbol,gD1DistCache[idx],gD1SlopeCache[idx],gD1SideDurCache[idx]);
      gD1LastBarTime[idx]=bt;
   }
   d1DistNorm=gD1DistCache[idx];
   d1SlopeNorm=gD1SlopeCache[idx];
   d1SideDurNorm=gD1SideDurCache[idx];
}

void UpdateBaseIndicatorsIfNewBar(const int symIdx)
{
   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym,BaseTF,0);
   if(bt==0) return;
   if(bt==gIndLastBarTime[symIdx]) return;

   double v;
   if(Copy1(hRSI_Base[symIdx],0,v))  gRSI_Base[symIdx]=v;
   if(Copy1(hCCI_Base[symIdx],0,v))  gCCI_Base[symIdx]=v;
   if(Copy1(hMACD_Base[symIdx],0,v)) gMACD_BaseMain[symIdx]=v;
   if(Copy1(hEMA_Base[symIdx],0,v))  gEMA_BaseVal[symIdx]=v;
   if(Copy1(hRVI_Base[symIdx],0,v))  gRVI_BaseMain[symIdx]=v;
   if(Copy1(hATRfast_Base[symIdx],0,v)) gATRfast_BaseVal[symIdx]=v;
   if(Copy1(hATRslow_Base[symIdx],0,v)) gATRslow_BaseVal[symIdx]=v;

   gIndLastBarTime[symIdx]=bt;
}

void UpdateH1IfNewBar(const int symIdx)
{
   if(!UseH1Features) return;

   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym,H1_TF,0);
   if(bt==0) return;
   if(bt==gH1LastBar[symIdx]) return;

   double v;
   if(Copy1(hRSI_H1[symIdx],0,v))  gRSI_H1v[symIdx]=v;
   if(Copy1(hMACD_H1[symIdx],0,v)) gMACD_H1v[symIdx]=v;
   if(Copy1(hEMA_H1[symIdx],0,v))  gEMA_H1v[symIdx]=v;
   if(Copy1(hRVI_H1[symIdx],0,v))  gRVI_H1v[symIdx]=v;

   double aF=0.0,aS=0.0;
   if(Copy1(hATRfast_H1[symIdx],0,aF) && Copy1(hATRslow_H1[symIdx],0,aS) && aS>0.0)
      gATRratio_H1[symIdx]=Clamp(aF/aS,0.1,5.0);

   gH1LastBar[symIdx]=bt;
}

void UpdateH4IfNewBar(const int symIdx)
{
   if(!UseH4Features) return;

   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym,H4_TF,0);
   if(bt==0) return;
   if(bt==gH4LastBar[symIdx]) return;

   double v;
   if(Copy1(hRSI_H4[symIdx],0,v))  gRSI_H4v[symIdx]=v;
   if(Copy1(hMACD_H4[symIdx],0,v)) gMACD_H4v[symIdx]=v;
   if(Copy1(hEMA_H4[symIdx],0,v))  gEMA_H4v[symIdx]=v;
   if(Copy1(hRVI_H4[symIdx],0,v))  gRVI_H4v[symIdx]=v;

   double aF=0.0,aS=0.0;
   if(Copy1(hATRfast_H4[symIdx],0,aF) && Copy1(hATRslow_H4[symIdx],0,aS) && aS>0.0)
      gATRratio_H4[symIdx]=Clamp(aF/aS,0.1,5.0);

   gH4LastBar[symIdx]=bt;
}


struct DDQNBranchLayout
{
   int basketStart;
   int basketCount;
   int indicatorStart;
   int indicatorCount;
   int volatilityStart;
   int volatilityCount;
   int structureStart;
   int structureCount;
   int zoneCandleStart;
   int zoneCandleCount;
   int totalCount;
};

DDQNBranchLayout gBranchLayout;

struct SlowFeatureCacheEntry
{
   bool valid;
   string symbol;
   datetime execBar;
   datetime midBar;
   datetime longBar;
   long avgQ;
   long lastEntryQ;
   double feats[];
};

SlowFeatureCacheEntry gIndicatorCache[MAX_SYMBOLS];
SlowFeatureCacheEntry gVolatilityCache[MAX_SYMBOLS];
SlowFeatureCacheEntry gStructureCache[MAX_SYMBOLS];
SlowFeatureCacheEntry gZoneCandleCache[MAX_SYMBOLS];

int gDecisionBarsSinceTrain = 0;

bool CacheMatchBasic(const SlowFeatureCacheEntry &c,const string symbol,const datetime execBar,const datetime midBar,const datetime longBar,const long avgQ,const long lastEntryQ,const bool needBasket)
{
   if(!c.valid) return false;
   if(c.symbol!=symbol) return false;
   if(c.execBar!=execBar || c.midBar!=midBar || c.longBar!=longBar) return false;
   if(needBasket)
   {
      if(c.avgQ!=avgQ || c.lastEntryQ!=lastEntryQ) return false;
   }
   return true;
}

void CloneDoubleArray(const double &src[], double &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}


double SiLU(const double x)
{
   if(x>=0.0)
   {
      double e=MathExp(-x);
      return x/(1.0+e);
   }
   else
   {
      double e=MathExp(x);
      return x*e/(1.0+e);
   }
}

double SiLUDerivativeFromPreAct(const double x)
{
   double s;
   if(x>=0.0)
   {
      double e=MathExp(-x);
      s = 1.0/(1.0+e);
   }
   else
   {
      double e=MathExp(x);
      s = e/(1.0+e);
   }
   return s + x*s*(1.0-s);
}

void InitBranchLayoutForStateDim(const int totalDim)
{
   gBranchLayout.basketStart     = 0;
   gBranchLayout.basketCount     = 0;
   gBranchLayout.indicatorStart  = 0;
   gBranchLayout.indicatorCount  = 0;
   gBranchLayout.volatilityStart = 0;
   gBranchLayout.volatilityCount = 0;
   gBranchLayout.structureStart  = 0;
   gBranchLayout.structureCount  = 0;
   gBranchLayout.zoneCandleStart = 0;
   gBranchLayout.zoneCandleCount = 0;
   gBranchLayout.totalCount      = MathMax(totalDim,0);

   int cursor=0;

   if(UseBasketBranch && UseStateV2SelfAwareness)
   {
      gBranchLayout.basketStart = cursor;
      gBranchLayout.basketCount = ModuleAFeatureCount();
      cursor += gBranchLayout.basketCount;
   }

   if(UseIndicatorBranch)
   {
      gBranchLayout.indicatorStart = cursor;
      gBranchLayout.indicatorCount = ModuleBFeatureCount();
      cursor += gBranchLayout.indicatorCount;
   }

   if(UseVolatilityBranch)
   {
      gBranchLayout.volatilityStart = cursor;
      gBranchLayout.volatilityCount = ModuleCFeatureCount();
      cursor += gBranchLayout.volatilityCount;
   }

   if(UseStructureBranch)
   {
      gBranchLayout.structureStart = cursor;
      gBranchLayout.structureCount = ModuleDFeatureCount();
      cursor += gBranchLayout.structureCount;
   }

   if(UseZoneCandleBranch)
   {
      gBranchLayout.zoneCandleStart = cursor;
      gBranchLayout.zoneCandleCount = ModuleEFeatureCount();
      cursor += gBranchLayout.zoneCandleCount;
   }

   // Safety remainder handling only for unexpected dimension drift.
   int rem = MathMax(totalDim - cursor, 0);
   if(rem > 0)
   {
      if(gBranchLayout.indicatorCount > 0)
         gBranchLayout.indicatorCount += rem;
      else
      {
         gBranchLayout.indicatorStart = cursor;
         gBranchLayout.indicatorCount = rem;
      }
      cursor += rem;
   }

   gBranchLayout.totalCount = cursor;
}

int ModuleAFeatureCount()
{
   return 31; // basket/self risk + memory/meta + pain-memory priors + Option B estimators + 3-mode regime scores
}

int ModuleBFeatureCount()
{
   return 22; // setup + macro-trend / reversal-trap factors
}

int ModuleCFeatureCount()
{
   return 16; // volatility / execution / regime-break factors + exact z-score risk guard
}

int ModuleDFeatureCount()
{
   return 18; // 3 TF * 6 structure-quality factors
}

int ModuleEFeatureCount()
{
   return 15; // 3 TF * 5 zone / candle factors
}

int StructureLookbackForTF(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_MID)  return MathMax(StructLookbackMid, 8);
   if(tf == TF_LONG) return MathMax(StructLookbackLong, 8);
   return MathMax(StructLookbackExec, 8);
}

int StructureHistoryBarsForTF(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_MID)  return MathMax(StructHistoryBarsMid, 200);
   if(tf == TF_LONG) return MathMax(StructHistoryBarsLong, 200);
   return MathMax(StructHistoryBarsExec, 200);
}

double StructNormDist(const double delta,const double atr)
{
   return Clamp(delta / MathMax(atr,1e-8), -StructDistClampAtr, StructDistClampAtr) / MathMax(StructDistClampAtr, 1.0);
}

double StructNormAmp(const double amp,const double atr)
{
   return Clamp(amp / MathMax(atr,1e-8), 0.0, StructAmpClampAtr) / MathMax(StructAmpClampAtr, 1.0);
}

struct SwingPointMem
{
   double   price;
   int      barShift;
   datetime when;
   bool     isHigh;
};

struct SwingCacheEntry
{
   bool valid;
   string symbol;
   ENUM_TIMEFRAMES tf;
   datetime barTime;
   SwingPointMem swings[];
   int swingCount;
   bool ok;
};

struct SwingDerivedZone
{
   bool     valid;
   bool     isDemand;
   double   low;
   double   high;
   double   center;
   double   displacementAtr;
   double   freshness;
   int      mitigationCount;
   bool     invalidated;
   int      lifeState;      // -1 invalidated, 0 candidate, +1 active
   int      pivotShift;
   datetime pivotTime;
};


struct ZoneCacheEntry
{
   bool valid;
   string symbol;
   ENUM_TIMEFRAMES tf;
   datetime barTime;
   SwingDerivedZone demands[];
   int demandCount;
   SwingDerivedZone supplies[];
   int supplyCount;
   bool ok;
};

SwingCacheEntry gSwingCache[];
ZoneCacheEntry  gZoneCache[];

struct StateCacheEntry
{
   bool valid;
   string symbol;
   datetime baseBarTime;
   int positionsCount;
   int basketDir;
   long avgQ;
   long lastEntryQ;
   double state[];
};

StateCacheEntry gStateCache[MAX_SYMBOLS];

void InitPerfCaches()
{
   int n=MathMax(8, SwingZoneCacheSlots);
   ArrayResize(gSwingCache, n);
   ArrayResize(gZoneCache, n);
   for(int i=0;i<n;i++)
   {
      gSwingCache[i].valid=false;
      gZoneCache[i].valid=false;
   }
   for(int i=0;i<MAX_SYMBOLS;i++)
      gStateCache[i].valid=false;
}

bool ShouldUseFastStateCache()
{
   if(!UseFastBacktestStateCache) return false;
   return (bool)MQLInfoInteger(MQL_TESTER);
}
void ZeroDoubleArray(double &arr[], const int count)
{
   int n=MathMax(count,0);
   ArrayResize(arr,n);
   for(int i=0;i<n;i++) arr[i]=0.0;
}

string BuildCurrentDQNMetadataNote()
{
   return StringFormat("shape=full_state; zero_fill_shape_compatible=1; reward_architecture=ddqn_risk_first_v5_archive_replay_symbol_local; archive_live_similarity=%d; archive_live_blend=%g; archive_add_risk_gate=%d; archive_add_risk_threshold=%g; replay_aware_reward=%d; replay_reward_scan=%d; replay_diag=%d; post_action_next_state=%d; symbol_local_budget_reward=%d; regime_warmup_only=%d; soft_regime_inference=%d; dd_event_bias=%d; dd_event_caution_only=%d; main_replay_persistence=%d; recent_replay_persistence=%d; branch_caps=%d_%d_%d; fusion_hidden=%d_%d; fast_training=%d; fast_mask_total=83_including_self; fast_mask_modules=B21_C9_D27_E20; skip_indicator=%d; skip_volatility=%d; skip_structure=%d; skip_zone_candle=%d; structure_zone_cached_closed_bar=1; dense_basket_feedback=%d; archive_aware_reward=%d; efficient_period_live_support=%d; min_add_spacing_gate=%d; dd_event_trace_on_bars=%d; dd_event_trace_tf=%d; replay_every_n_decision_bars=%d; replay_train_iters=%d; replay_batch_samples=%d",
                       (UseArchiveLiveSimilarity ? 1 : 0),
                       ArchiveLiveBlendWeight,
                       (UseArchiveAddRiskGate ? 1 : 0),
                       ArchiveAddRiskGateThreshold,
                       (UseReplayAwareReward ? 1 : 0),
                       MathMax(50, ReplayRewardScanLimit),
                       (UseReplayDiagnostics ? 1 : 0),
                       (UseRealPostActionTransitions ? 1 : 0),
                       (UsePerSymbolVirtualBudget ? 1 : 0),
                       (TrainAllRegimesWarmupOnly ? 1 : 0),
                       (UseSoftRegimeInference ? 1 : 0),
                       (UseDDEventBias ? 1 : 0),
                       (DDEventUseCautionOnly ? 1 : 0),
                       (SaveMainReplayBank ? 1 : 0),
                       (SaveRecentReplayBank ? 1 : 0),
                       BranchEncoderMinWidth,
                       BranchEncoderH1Cap,
                       BranchEncoderH2Cap,
                       HiddenSize,
                       HiddenSize2,
                       (FastTrainingMode ? 1 : 0),
                       (FastTrainingSkipIndicatorBranch ? 1 : 0),
                       (FastTrainingSkipVolatilityBranch ? 1 : 0),
                       (FastTrainingSkipStructureBranch ? 1 : 0),
                       (FastTrainingSkipZoneCandleBranch ? 1 : 0),
                       (UseDenseBasketHealthReward ? 1 : 0),
                       (UseArchiveAwareReward ? 1 : 0),
                       (UseEfficientPeriodLiveSupport ? 1 : 0),
                       (UseMinAddSpacingGate ? 1 : 0),
                       (DDEventTraceOnBars ? 1 : 0),
                       (int)DDEventTraceTFForSymbol(_Symbol),
                       MathMax(1, TrainEveryNDecisionBars),
                       MathMax(1, ReplayTrainIters),
                       MathMax(1, ReplayBatchSize));
}

bool ShouldTrainReplayBatchNow(const int symIdx,const string symbol)
{
   if(!UseReplayBuffer || !isTraining) return false;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return false;
   if(ArraySize(gReplay) < ReplayWarmup || ReplayBatchSize<=0) return false;
   if(gReplayPendingTrainCount <= 0) return false;

   datetime barTime=iTime(symbol, BaseTF, 1);
   if(barTime<=0) return false;
   if(gLastReplayDecisionBarTime[symIdx]==barTime) return false;

   gLastReplayDecisionBarTime[symIdx]=barTime;
   gReplayDecisionBarCounter[symIdx]++;

   int everyN=MathMax(1, TrainEveryNDecisionBars);
   return ((gReplayDecisionBarCounter[symIdx] % everyN) == 0);
}

long QuantizePriceByPoint(const double price,const double point)
{
   double p=(point>0.0?point:0.00001);
   return (long)MathRound(price/p);
}

void CloneSwingArray(const SwingPointMem &src[], SwingPointMem &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}

void CloneZoneArray(const SwingDerivedZone &src[], SwingDerivedZone &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}

int FindOrAllocSwingCacheSlot(const string symbol,const ENUM_TIMEFRAMES tf)
{
   int freeIdx=-1;
   for(int i=0;i<ArraySize(gSwingCache);i++)
   {
      if(gSwingCache[i].valid && gSwingCache[i].symbol==symbol && gSwingCache[i].tf==tf)
         return i;
      if(!gSwingCache[i].valid && freeIdx<0)
         freeIdx=i;
   }
   if(freeIdx>=0) return freeIdx;
   return 0;
}

int FindOrAllocZoneCacheSlot(const string symbol,const ENUM_TIMEFRAMES tf)
{
   int freeIdx=-1;
   for(int i=0;i<ArraySize(gZoneCache);i++)
   {
      if(gZoneCache[i].valid && gZoneCache[i].symbol==symbol && gZoneCache[i].tf==tf)
         return i;
      if(!gZoneCache[i].valid && freeIdx<0)
         freeIdx=i;
   }
   if(freeIdx>=0) return freeIdx;
   return 0;
}

void AppendSwingMem(SwingPointMem &arr[], int &count, const int maxCount, const double price, const int barShift, const datetime when, const bool isHigh)
{
   if(maxCount <= 0) return;
   if(count < maxCount)
   {
      ArrayResize(arr, count + 1);
      arr[count].price    = price;
      arr[count].barShift = barShift;
      arr[count].when     = when;
      arr[count].isHigh   = isHigh;
      count++;
      return;
   }

   for(int i=1; i<count; ++i)
      arr[i-1] = arr[i];

   arr[count-1].price    = price;
   arr[count-1].barShift = barShift;
   arr[count-1].when     = when;
   arr[count-1].isHigh   = isHigh;
}

bool BuildSwingMemoryForTF_Core(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           SwingPointMem &swings[],
                           int &swingCount)
{
   swingCount = 0;
   ArrayResize(swings, 0);

   int barsWanted = StructureHistoryBarsForTF(tf);
   int need = MathMax(barsWanted, 120);
   if(Bars(symbol, tf) < need + 5)
      return false;

   double highs[], lows[], closes[], atrs[];
   datetime times[];
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(atrs, true);
   ArraySetAsSeries(times, true);

   int copiedH = CopyHigh(symbol, tf, 1, need, highs);
   int copiedL = CopyLow(symbol, tf, 1, need, lows);
   int copiedC = CopyClose(symbol, tf, 1, need, closes);
   int copiedT = CopyTime(symbol, tf, 1, need, times);

   int atrHandle = iATR(symbol, tf, 14);
   int copiedA = 0;
   if(atrHandle != INVALID_HANDLE)
   {
      copiedA = CopyBuffer(atrHandle, 0, 1, need, atrs);
      IndicatorRelease(atrHandle);
   }

   int n = MathMin(MathMin(copiedH, copiedL), MathMin(copiedC, copiedT));
   if(copiedA > 0) n = MathMin(n, copiedA);
   if(n < 40)
      return false;

   int oldest = n - 1;
   int mode = 0; // +1 up leg, -1 down leg, 0 unknown

   if(closes[oldest - 1] > closes[oldest]) mode = +1;
   else if(closes[oldest - 1] < closes[oldest]) mode = -1;

   double candidateHigh = highs[oldest];
   double candidateLow  = lows[oldest];
   int    candidateHighShift = oldest + 1; // because arrays start at shift=1
   int    candidateLowShift  = oldest + 1;
   datetime candidateHighTime = times[oldest];
   datetime candidateLowTime  = times[oldest];

   for(int i = oldest - 1; i >= 0; --i)
   {
      double atr = (i < ArraySize(atrs) ? atrs[i] : 0.0);
      if(atr <= 1e-8)
      {
         double close_i = closes[i];
         atr = MathMax(MathAbs(candidateHigh - close_i), MathAbs(close_i - candidateLow));
         atr = MathMax(atr, 1e-6);
      }
      double reversal = MathMax(atr * SwingReversalAtrMult, _Point * 10.0);

      if(mode >= 0)
      {
         if(highs[i] >= candidateHigh)
         {
            candidateHigh = highs[i];
            candidateHighShift = i + 1;
            candidateHighTime  = times[i];
         }

         if((candidateHigh - lows[i]) >= reversal)
         {
            AppendSwingMem(swings, swingCount, StructMaxSwingsPerTF, candidateHigh, candidateHighShift, candidateHighTime, true);
            mode = -1;
            candidateLow = lows[i];
            candidateLowShift = i + 1;
            candidateLowTime  = times[i];
            continue;
         }
      }

      if(mode <= 0)
      {
         if(lows[i] <= candidateLow)
         {
            candidateLow = lows[i];
            candidateLowShift = i + 1;
            candidateLowTime  = times[i];
         }

         if((highs[i] - candidateLow) >= reversal)
         {
            AppendSwingMem(swings, swingCount, StructMaxSwingsPerTF, candidateLow, candidateLowShift, candidateLowTime, false);
            mode = +1;
            candidateHigh = highs[i];
            candidateHighShift = i + 1;
            candidateHighTime  = times[i];
            continue;
         }
      }
   }

   
return (swingCount > 0);
}

bool BuildSwingMemoryForTF(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           SwingPointMem &swings[],
                           int &swingCount)
{
   swingCount=0;
   ArrayResize(swings,0);

   datetime barTime=iTime(symbol, tf, 1);
   int slot=FindOrAllocSwingCacheSlot(symbol, tf);
   if(slot>=0 && gSwingCache[slot].valid && gSwingCache[slot].symbol==symbol && gSwingCache[slot].tf==tf && gSwingCache[slot].barTime==barTime)
   {
      swingCount=gSwingCache[slot].swingCount;
      CloneSwingArray(gSwingCache[slot].swings, swings);
      return gSwingCache[slot].ok;
   }

   bool ok=BuildSwingMemoryForTF_Core(symbol, tf, swings, swingCount);

   if(slot>=0)
   {
      gSwingCache[slot].valid=true;
      gSwingCache[slot].symbol=symbol;
      gSwingCache[slot].tf=tf;
      gSwingCache[slot].barTime=barTime;
      gSwingCache[slot].swingCount=swingCount;
      gSwingCache[slot].ok=ok;
      CloneSwingArray(swings, gSwingCache[slot].swings);
   }
   return ok;
}


bool BuildSwingDerivedZonesForTF(const string symbol,
                                 const ENUM_TIMEFRAMES tf,
                                 SwingDerivedZone &demands[],
                                 int &demandCount,
                                 SwingDerivedZone &supplies[],
                                 int &supplyCount)
{
   demandCount = 0;
   supplyCount = 0;
   ArrayResize(demands, 0);
   ArrayResize(supplies, 0);

   datetime barTime=iTime(symbol, tf, 1);
   int slot=FindOrAllocZoneCacheSlot(symbol, tf);
   if(slot>=0 && gZoneCache[slot].valid && gZoneCache[slot].symbol==symbol && gZoneCache[slot].tf==tf && gZoneCache[slot].barTime==barTime)
   {
      demandCount=gZoneCache[slot].demandCount;
      supplyCount=gZoneCache[slot].supplyCount;
      CloneZoneArray(gZoneCache[slot].demands, demands);
      CloneZoneArray(gZoneCache[slot].supplies, supplies);
      return gZoneCache[slot].ok;
   }

   bool ok=BuildSwingDerivedZonesForTF_Core(symbol, tf, demands, demandCount, supplies, supplyCount);

   if(slot>=0)
   {
      gZoneCache[slot].valid=true;
      gZoneCache[slot].symbol=symbol;
      gZoneCache[slot].tf=tf;
      gZoneCache[slot].barTime=barTime;
      gZoneCache[slot].demandCount=demandCount;
      gZoneCache[slot].supplyCount=supplyCount;
      gZoneCache[slot].ok=ok;
      CloneZoneArray(demands, gZoneCache[slot].demands);
      CloneZoneArray(supplies, gZoneCache[slot].supplies);
   }
   return ok;
}

bool ExtractRecentSwingRefs(const SwingPointMem &swings[],
                            const int swingCount,
                            SwingPointMem &lastHigh,
                            SwingPointMem &prevHigh,
                            SwingPointMem &lastLow,
                            SwingPointMem &prevLow)
{
   bool hasLastHigh=false, hasPrevHigh=false, hasLastLow=false, hasPrevLow=false;
   for(int i=swingCount-1; i>=0; --i)
   {
      if(swings[i].isHigh)
      {
         if(!hasLastHigh) { lastHigh = swings[i]; hasLastHigh = true; }
         else if(!hasPrevHigh) { prevHigh = swings[i]; hasPrevHigh = true; }
      }
      else
      {
         if(!hasLastLow) { lastLow = swings[i]; hasLastLow = true; }
         else if(!hasPrevLow) { prevLow = swings[i]; hasPrevLow = true; }
      }
      if(hasLastHigh && hasPrevHigh && hasLastLow && hasPrevLow)
         break;
   }
   return (hasLastHigh || hasLastLow);
}

void FallbackStructureRefs(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           SwingPointMem &lastHigh,
                           SwingPointMem &prevHigh,
                           SwingPointMem &lastLow,
                           SwingPointMem &prevLow)
{
   int lookback = StructureLookbackForTF(tf);
   lastHigh.price = HighestInRangeTF(symbol, tf, 1, lookback);
   lastLow.price  = LowestInRangeTF(symbol, tf, 1, lookback);
   prevHigh.price = HighestInRangeTF(symbol, tf, lookback + 1, lookback);
   prevLow.price  = LowestInRangeTF(symbol, tf, lookback + 1, lookback);

   lastHigh.barShift = HighestIndexInRangeTF(symbol, tf, 1, lookback);
   lastLow.barShift  = LowestIndexInRangeTF(symbol, tf, 1, lookback);
   prevHigh.barShift = HighestIndexInRangeTF(symbol, tf, lookback + 1, lookback);
   prevLow.barShift  = LowestIndexInRangeTF(symbol, tf, lookback + 1, lookback);

   lastHigh.when = iTime(symbol, tf, lastHigh.barShift);
   lastLow.when  = iTime(symbol, tf, lastLow.barShift);
   prevHigh.when = iTime(symbol, tf, prevHigh.barShift);
   prevLow.when  = iTime(symbol, tf, prevLow.barShift);

   lastHigh.isHigh = true;
   prevHigh.isHigh = true;
   lastLow.isHigh  = false;
   prevLow.isHigh  = false;
}

double SwingTendencyFromMemory(const SwingPointMem &lastHigh,
                               const SwingPointMem &prevHigh,
                               const SwingPointMem &lastLow,
                               const SwingPointMem &prevLow,
                               const double atr)
{
   double highProg = Clamp((lastHigh.price - prevHigh.price) / MathMax(atr,1e-8), -StructDistClampAtr, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
   double lowProg  = Clamp((lastLow.price - prevLow.price) / MathMax(atr,1e-8), -StructDistClampAtr, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
   return Clamp(0.5 * (highProg + lowProg), -1.0, 1.0);
}

double SwingSequenceStrengthFromMemory(const SwingPointMem &lastHigh,
                                       const SwingPointMem &prevHigh,
                                       const SwingPointMem &lastLow,
                                       const SwingPointMem &prevLow,
                                       const double atr,
                                       const double tendency)
{
   double seqStrengthRaw = 0.5 * (MathAbs(lastHigh.price - prevHigh.price) + MathAbs(lastLow.price - prevLow.price));
   double seqStrength = StructNormAmp(seqStrengthRaw, atr);
   if(tendency < 0.0) seqStrength = -seqStrength;
   return seqStrength;
}



double BullBOSStrengthFromMemory(const double price,
                                 const SwingPointMem &lastHigh,
                                 const double atr,
                                 const double tendency)
{
   if(price <= lastHigh.price || tendency < -0.05)
      return 0.0;
   return Clamp((price - lastHigh.price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double BearBOSStrengthFromMemory(const double price,
                                 const SwingPointMem &lastLow,
                                 const double atr,
                                 const double tendency)
{
   if(price >= lastLow.price || tendency > 0.05)
      return 0.0;
   return Clamp((lastLow.price - price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double BullCHoCHStrengthFromMemory(const double price,
                                   const SwingPointMem &lastHigh,
                                   const double atr,
                                   const double tendency)
{
   if(price <= lastHigh.price || tendency >= 0.05)
      return 0.0;
   return Clamp((price - lastHigh.price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double BearCHoCHStrengthFromMemory(const double price,
                                   const SwingPointMem &lastLow,
                                   const double atr,
                                   const double tendency)
{
   if(price >= lastLow.price || tendency <= -0.05)
      return 0.0;
   return Clamp((lastLow.price - price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double StrongWeakHighStateFromMemory(const double price,
                                     const SwingPointMem &lastHigh,
                                     const double atr,
                                     const double tendency,
                                     const double seqStrength)
{
   double proxHigh = 1.0 - Clamp(MathAbs(lastHigh.price - price) / MathMax(atr * MathMax(StructDistClampAtr,1.0), 1e-8), 0.0, 1.0);
   double weakScore = Clamp(0.60 * MathMax(tendency, 0.0) + 0.20 * MathMax(seqStrength, 0.0) + 0.20 * proxHigh, 0.0, 1.0);
   return Clamp(1.0 - 2.0 * weakScore, -1.0, 1.0); // +1 strong, -1 weak
}

double StrongWeakLowStateFromMemory(const double price,
                                    const SwingPointMem &lastLow,
                                    const double atr,
                                    const double tendency,
                                    const double seqStrength)
{
   double proxLow = 1.0 - Clamp(MathAbs(price - lastLow.price) / MathMax(atr * MathMax(StructDistClampAtr,1.0), 1e-8), 0.0, 1.0);
   double weakScore = Clamp(0.60 * MathMax(-tendency, 0.0) + 0.20 * MathMax(-seqStrength, 0.0) + 0.20 * proxLow, 0.0, 1.0);
   return Clamp(1.0 - 2.0 * weakScore, -1.0, 1.0); // +1 strong, -1 weak
}
struct ZoneBranchFeaturePack
{
   double nearestDemandDist;
   double nearestSupplyDist;
   double zoneSideState;
   double zoneDepthPosition;
   double zoneFreshness;
   double zoneMitigation;
   double zoneInvalidationState;
   double zoneRelevance;
   double zoneRetestBreakState;
};

void InitZoneBranchFeaturePack(ZoneBranchFeaturePack &z)
{
   z.nearestDemandDist=0.0;
   z.nearestSupplyDist=0.0;
   z.zoneSideState=0.0;
   z.zoneDepthPosition=0.0;
   z.zoneFreshness=0.0;
   z.zoneMitigation=0.0;
   z.zoneInvalidationState=0.0;
   z.zoneRelevance=0.0;
   z.zoneRetestBreakState=0.0;
}


void InitSwingDerivedZone(SwingDerivedZone &z)
{
   z.valid=false;
   z.isDemand=false;
   z.low=0.0;
   z.high=0.0;
   z.center=0.0;
   z.displacementAtr=0.0;
   z.freshness=0.0;
   z.mitigationCount=0;
   z.invalidated=false;
   z.lifeState=0;
   z.pivotShift=0;
   z.pivotTime=0;
}

double ZoneFreshnessFromShift(const ENUM_TIMEFRAMES tf,const int pivotShift)
{
   int barsBase = MathMax(StructureHistoryBarsForTF(tf), 200);
   return 1.0 - Clamp((double)pivotShift / (double)barsBase, 0.0, 1.0);
}

void GetPivotBodyBounds(const string symbol,
                        const ENUM_TIMEFRAMES tf,
                        const int pivotShift,
                        const int clusterBars,
                        double &bodyLow,
                        double &bodyHigh)
{
   bodyLow = DBL_MAX;
   bodyHigh = -DBL_MAX;
   int left = MathMax(clusterBars, 0);
   for(int d=-left; d<=left; ++d)
   {
      int sh = pivotShift + d;
      if(sh < 1) continue;
      double o = iOpen(symbol, tf, sh);
      double c = GetCloseSafe(symbol, tf, sh);
      double lo = MathMin(o, c);
      double hi = MathMax(o, c);
      if(lo < bodyLow) bodyLow = lo;
      if(hi > bodyHigh) bodyHigh = hi;
   }
   if(bodyLow >= DBL_MAX/2.0) bodyLow = MathMin(iOpen(symbol, tf, pivotShift), GetCloseSafe(symbol, tf, pivotShift));
   if(bodyHigh <= -DBL_MAX/2.0) bodyHigh = MathMax(iOpen(symbol, tf, pivotShift), GetCloseSafe(symbol, tf, pivotShift));
}

void FinalizeZoneTouches(const string symbol,
                         const ENUM_TIMEFRAMES tf,
                         SwingDerivedZone &z)
{
   if(!z.valid) return;
   double atr = MathMax(GetATRValueTF(symbol, tf, 14, z.pivotShift), 1e-8);
   double buffer = MathMax(MathAbs(ZoneInvalidationBufferAtr) * atr, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);

   z.mitigationCount = 0;
   z.invalidated = false;
   z.lifeState = 0;

   for(int sh = z.pivotShift - 1; sh >= 1; --sh)
   {
      double hi = iHigh(symbol, tf, sh);
      double lo = iLow(symbol, tf, sh);
      double cl = GetCloseSafe(symbol, tf, sh);

      bool touched = (hi >= z.low && lo <= z.high);
      if(touched)
         z.mitigationCount++;

      if(z.isDemand)
      {
         if(cl < z.low - buffer)
         {
            z.invalidated = true;
            break;
         }
      }
      else
      {
         if(cl > z.high + buffer)
         {
            z.invalidated = true;
            break;
         }
      }
   }

   if(z.invalidated) z.lifeState = -1;
   else if(z.mitigationCount > 0) z.lifeState = +1;
   else z.lifeState = 0;
}

bool BuildSwingDerivedZonesForTF_Core(const string symbol,
                                 const ENUM_TIMEFRAMES tf,
                                 SwingDerivedZone &demands[],
                                 int &demandCount,
                                 SwingDerivedZone &supplies[],
                                 int &supplyCount)
{
   demandCount = 0;
   supplyCount = 0;
   ArrayResize(demands, 0);
   ArrayResize(supplies, 0);

   SwingPointMem swings[];
   int swingCount = 0;
   if(!BuildSwingMemoryForTF_Core(symbol, tf, swings, swingCount) || swingCount < 2)
      return false;

   int startIdx = MathMax(0, swingCount - MathMax(ZoneMaxLookbackSwings, 4));
   for(int i=startIdx; i < swingCount - 1; ++i)
   {
      SwingPointMem a = swings[i];
      SwingPointMem b = swings[i+1];
      double atrA = MathMax(GetATRValueTF(symbol, tf, 14, a.barShift), 1e-8);

      if(!a.isHigh && b.isHigh)
      {
         double disp = (b.price - a.price) / atrA;
         if(disp >= ZoneMinDisplacementAtr)
         {
            SwingDerivedZone z;
            InitSwingDerivedZone(z);
            z.valid = true;
            z.isDemand = true;
            z.pivotShift = a.barShift;
            z.pivotTime  = a.when;
            z.displacementAtr = disp;
            z.freshness = ZoneFreshnessFromShift(tf, a.barShift);

            double bodyLow, bodyHigh;
            GetPivotBodyBounds(symbol, tf, a.barShift, ZonePivotClusterBars, bodyLow, bodyHigh);
            z.low = a.price;
            z.high = MathMax(bodyLow, z.low + SymbolInfoDouble(symbol,SYMBOL_POINT) * 2.0);
            z.center = 0.5 * (z.low + z.high);

            FinalizeZoneTouches(symbol, tf, z);
            ArrayResize(demands, demandCount + 1);
            demands[demandCount++] = z;
         }
      }
      else if(a.isHigh && !b.isHigh)
      {
         double disp = (a.price - b.price) / atrA;
         if(disp >= ZoneMinDisplacementAtr)
         {
            SwingDerivedZone z;
            InitSwingDerivedZone(z);
            z.valid = true;
            z.isDemand = false;
            z.pivotShift = a.barShift;
            z.pivotTime  = a.when;
            z.displacementAtr = disp;
            z.freshness = ZoneFreshnessFromShift(tf, a.barShift);

            double bodyLow, bodyHigh;
            GetPivotBodyBounds(symbol, tf, a.barShift, ZonePivotClusterBars, bodyLow, bodyHigh);
            z.high = a.price;
            z.low = MathMin(bodyHigh, z.high - SymbolInfoDouble(symbol,SYMBOL_POINT) * 2.0);
            z.center = 0.5 * (z.low + z.high);

            FinalizeZoneTouches(symbol, tf, z);
            ArrayResize(supplies, supplyCount + 1);
            supplies[supplyCount++] = z;
         }
      }
   }

   return (demandCount > 0 || supplyCount > 0);
}



double DistanceToZoneBoundaryAtr(const SwingDerivedZone &z,const double px,const double atr)
{
   double a = MathMax(atr, 1e-8);
   if(px < z.low)  return (z.low - px) / a;
   if(px > z.high) return (px - z.high) / a;
   return 0.0;
}

double ZoneDepthForPrice(const SwingDerivedZone &z,const double px,const double atr)
{
   double h = MathMax(z.high - z.low, MathMax(atr,1e-8) * 0.10);
   double buf = MathMax(MathAbs(ZoneNearBufferAtr) * MathMax(atr,1e-8), SymbolInfoDouble(_Symbol,SYMBOL_POINT) * 5.0);
   if(px < z.low - buf || px > z.high + buf)
      return 0.0;

   if(z.isDemand)
   {
      double deepRef = z.low - buf;
      double shallowRef = z.high + buf;
      double depth = 1.0 - Clamp((px - deepRef) / MathMax(shallowRef - deepRef, 1e-8), 0.0, 1.0);
      return (depth * 2.0) - 1.0;
   }
   else
   {
      double shallowRef = z.low - buf;
      double deepRef = z.high + buf;
      double depth = Clamp((px - shallowRef) / MathMax(deepRef - shallowRef, 1e-8), 0.0, 1.0);
      return (depth * 2.0) - 1.0;
   }
}

double ZoneStateWeight(const SwingDerivedZone &z)
{
   if(z.lifeState > 0)  return 1.00; // active
   if(z.lifeState == 0) return 0.72; // candidate / first-touch
   return 0.30;                      // invalidated
}

bool IsTFEnabledForStructure(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_EXEC)       return UseTFExecFeatures;
   if(tf == TF_MID)        return UseTFMidFeatures;
   if(tf == TF_LONG)       return UseTFLongFeatures;
   if(tf == TF_STRUCT_EXT) return UseTFStructExtFeatures;
   return true;
}

double TFImportanceWeight(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_EXEC)       return 1.00;
   if(tf == TF_MID)        return 1.10;
   if(tf == TF_LONG)       return 1.20;
   if(tf == TF_STRUCT_EXT) return 1.30;
   return 1.00;
}

double ZoneOverlapDistanceAtr(const SwingDerivedZone &a,const SwingDerivedZone &b,const double atr)
{
   double lo = MathMax(a.low, b.low);
   double hi = MathMin(a.high, b.high);
   if(lo <= hi) return 0.0;
   if(a.high < b.low) return (b.low - a.high) / MathMax(atr,1e-8);
   return (a.low - b.high) / MathMax(atr,1e-8);
}

double ZoneConfluenceBoost(const string symbol,
                           const ENUM_TIMEFRAMES baseTf,
                           const SwingDerivedZone &z,
                           const double atr)
{
   ENUM_TIMEFRAMES tfs[4] = { TF_EXEC, TF_MID, TF_LONG, TF_STRUCT_EXT };
   double boost = 1.0;
   for(int i=0;i<4;i++)
   {
      ENUM_TIMEFRAMES tf2 = tfs[i];
      if(tf2 == baseTf) continue;
      if(!IsTFEnabledForStructure(tf2)) continue;

      SwingDerivedZone demands2[], supplies2[];
      int dc2=0, sc2=0;
      if(!BuildSwingDerivedZonesForTF(symbol, tf2, demands2, dc2, supplies2, sc2))
         continue;

      SwingDerivedZone bestOther;
      bool found = false;
      double bestDist = DBL_MAX;
      if(z.isDemand)
      {
         for(int j=0;j<dc2;j++)
         {
            if(!demands2[j].valid) continue;
            double d = ZoneOverlapDistanceAtr(z, demands2[j], atr);
            if(d < bestDist) { bestDist = d; bestOther = demands2[j]; found = true; }
         }
      }
      else
      {
         for(int j=0;j<sc2;j++)
         {
            if(!supplies2[j].valid) continue;
            double d = ZoneOverlapDistanceAtr(z, supplies2[j], atr);
            if(d < bestDist) { bestDist = d; bestOther = supplies2[j]; found = true; }
         }
      }

      if(found && bestDist <= MathAbs(ZoneConfluenceAtr))
      {
         double closeness = 1.0 - Clamp(bestDist / MathMax(MathAbs(ZoneConfluenceAtr),1e-8), 0.0, 1.0);
         boost += 0.12 * TFImportanceWeight(tf2) * closeness * ZoneStateWeight(bestOther);
      }
   }
   return Clamp(boost, 1.0, 1.6);
}

double SwingLevelConfluenceBoost(const string symbol,
                                 const ENUM_TIMEFRAMES baseTf,
                                 const double level,
                                 const bool isHigh,
                                 const double atr)
{
   ENUM_TIMEFRAMES tfs[4] = { TF_EXEC, TF_MID, TF_LONG, TF_STRUCT_EXT };
   double boost = 1.0;
   for(int i=0;i<4;i++)
   {
      ENUM_TIMEFRAMES tf2 = tfs[i];
      if(tf2 == baseTf) continue;
      if(!IsTFEnabledForStructure(tf2)) continue;

      SwingPointMem swings2[];
      int count2 = 0;
      if(!BuildSwingMemoryForTF(symbol, tf2, swings2, count2) || count2 < 1)
         continue;

      double best = DBL_MAX;
      for(int j=count2-1; j>=0 && j>=count2-12; --j)
      {
         if(swings2[j].isHigh != isHigh) continue;
         double d = MathAbs(swings2[j].price - level) / MathMax(atr,1e-8);
         if(d < best) best = d;
      }
      if(best <= MathAbs(SwingConfluenceAtr))
      {
         double closeness = 1.0 - Clamp(best / MathMax(MathAbs(SwingConfluenceAtr),1e-8), 0.0, 1.0);
         boost += 0.10 * TFImportanceWeight(tf2) * closeness;
      }
   }
   return Clamp(boost, 1.0, 1.5);
}

double SwingNearnessScore(const string symbol,const ENUM_TIMEFRAMES tf,const double px,const double atr)
{
   SwingPointMem swings[];
   int swingCount = 0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   bool ok = BuildSwingMemoryForTF(symbol, tf, swings, swingCount);
   ZeroMemory(lastHigh); ZeroMemory(prevHigh); ZeroMemory(lastLow); ZeroMemory(prevLow);
   bool hasRefs = false;
   if(ok)
      hasRefs = ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);
   if(!hasRefs)
      FallbackStructureRefs(symbol, tf, lastHigh, prevHigh, lastLow, prevLow);

   double d = DBL_MAX;
   d = MathMin(d, MathAbs(px - lastHigh.price) / MathMax(atr,1e-8));
   d = MathMin(d, MathAbs(px - lastLow.price) / MathMax(atr,1e-8));
   d = MathMin(d, MathAbs(px - prevHigh.price) / MathMax(atr,1e-8));
   d = MathMin(d, MathAbs(px - prevLow.price) / MathMax(atr,1e-8));
   double nearBuf = MathMax(MathAbs(ZoneNearBufferAtr), 0.25);
   return 1.0 - Clamp(d / nearBuf, 0.0, 1.0);
}

double ZoneProximityScore(const SwingDerivedZone &z,const double px,const double avgPx,const double entryPx,const double atr)
{
   double wPrice = 0.50, wAvg = 0.30, wEntry = 0.20;

   double d1 = DistanceToZoneBoundaryAtr(z, px, atr);
   double d2 = DistanceToZoneBoundaryAtr(z, avgPx, atr);
   double d3 = DistanceToZoneBoundaryAtr(z, entryPx, atr);

   if(px >= z.low && px <= z.high)
      d1 = 0.0;
   if(avgPx >= z.low && avgPx <= z.high)
      d2 = 0.0;
   if(entryPx >= z.low && entryPx <= z.high)
      d3 = 0.0;

   double distAtr = (wPrice*d1 + wAvg*d2 + wEntry*d3);
   return 1.0 - Clamp(distAtr / 4.0, 0.0, 1.0);
}

bool SelectBestSwingZone(const SwingDerivedZone &zonesArr[],
                         const int count,
                         const string symbol,
                         const ENUM_TIMEFRAMES tf,
                         const double px,
                         const double avgPx,
                         const double entryPx,
                         const double atr,
                         int &bestIdx)
{
   bestIdx = -1;
   double bestScore = -DBL_MAX;
   for(int i=0; i<count; ++i)
   {
      if(!zonesArr[i].valid) continue;

      double prox = ZoneProximityScore(zonesArr[i], px, avgPx, entryPx, atr);
      double dispW = (0.65 + 0.35 * Clamp(zonesArr[i].displacementAtr / 3.0, 0.0, 1.0));
      double freshW = (0.55 + 0.45 * zonesArr[i].freshness);
      double mitiPenalty = (1.0 - 0.10 * MathMin(zonesArr[i].mitigationCount, 5));
      double stateW = ZoneStateWeight(zonesArr[i]);
      double confW = ZoneConfluenceBoost(symbol, tf, zonesArr[i], atr);

      double score = prox * dispW * freshW * mitiPenalty * stateW * confW;
      if(score > bestScore)
      {
         bestScore = score;
         bestIdx = i;
      }
   }
   return (bestIdx >= 0);
}

double ZoneRetestBreakStateForTF(const SwingDerivedZone &z,const string symbol,const ENUM_TIMEFRAMES tf,const double atr)
{
   double close1 = GetCloseSafe(symbol, tf, 1);
   double high1  = iHigh(symbol, tf, 1);
   double low1   = iLow(symbol, tf, 1);
   double open1  = iOpen(symbol, tf, 1);
   double buffer = MathMax(MathAbs(ZoneInvalidationBufferAtr) * MathMax(atr,1e-8), SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);

   if(z.isDemand)
   {
      if(close1 < z.low - buffer) return -1.0;
      if(low1 <= z.high + buffer && close1 > open1 && close1 >= z.low) return 0.5;
      if(low1 <= z.high + buffer) return 0.2;
      return 0.0;
   }
   else
   {
      if(close1 > z.high + buffer) return 1.0;
      if(high1 >= z.low - buffer && close1 < open1 && close1 <= z.high) return -0.5;
      if(high1 >= z.low - buffer) return -0.2;
      return 0.0;
   }
}

void ComputeZoneBranchFeatures(const string symbol,
                               const ENUM_TIMEFRAMES tf,
                               const double refPrice,
                               const double avgPrice,
                               const double lastEntryPrice,
                               ZoneBranchFeaturePack &outZ)
{
   InitZoneBranchFeaturePack(outZ);
   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);

   SwingDerivedZone demands[], supplies[];
   int demandCount = 0, supplyCount = 0;
   if(!BuildSwingDerivedZonesForTF(symbol, tf, demands, demandCount, supplies, supplyCount))
      return;

   int idxDem = -1, idxSup = -1;
   bool hasDem = SelectBestSwingZone(demands, demandCount, symbol, tf, refPrice, avgPrice, lastEntryPrice, atr, idxDem);
   bool hasSup = SelectBestSwingZone(supplies, supplyCount, symbol, tf, refPrice, avgPrice, lastEntryPrice, atr, idxSup);

   if(hasDem)
      outZ.nearestDemandDist = Clamp(DistanceToZoneBoundaryAtr(demands[idxDem], refPrice, atr), 0.0, 5.0) / 5.0;
   if(hasSup)
      outZ.nearestSupplyDist = Clamp(DistanceToZoneBoundaryAtr(supplies[idxSup], refPrice, atr), 0.0, 5.0) / 5.0;

   int activeSide = 0;
   if(hasDem && hasSup)
   {
      double sd = ZoneProximityScore(demands[idxDem], refPrice, avgPrice, lastEntryPrice, atr) * ZoneStateWeight(demands[idxDem]);
      double ss = ZoneProximityScore(supplies[idxSup], refPrice, avgPrice, lastEntryPrice, atr) * ZoneStateWeight(supplies[idxSup]);
      activeSide = (sd >= ss ? +1 : -1);
   }
   else if(hasDem) activeSide = +1;
   else if(hasSup) activeSide = -1;
   else return;

   SwingDerivedZone z = (activeSide > 0 ? demands[idxDem] : supplies[idxSup]);

   double proxPrice = 1.0 - Clamp(DistanceToZoneBoundaryAtr(z, refPrice, atr) / MathMax(MathAbs(ZoneNearBufferAtr), 0.25), 0.0, 1.0);
   double proxAvg   = 1.0 - Clamp(DistanceToZoneBoundaryAtr(z, avgPrice, atr) / MathMax(MathAbs(ZoneNearBufferAtr), 0.25), 0.0, 1.0);
   double proxEntry = 1.0 - Clamp(DistanceToZoneBoundaryAtr(z, lastEntryPrice, atr) / MathMax(MathAbs(ZoneNearBufferAtr), 0.25), 0.0, 1.0);
   double combinedProx = Clamp(0.50*proxPrice + 0.30*proxAvg + 0.20*proxEntry, 0.0, 1.0);

   outZ.zoneSideState = (z.isDemand ? 1.0 : -1.0);
   outZ.zoneDepthPosition = ZoneDepthForPrice(z, refPrice, atr);
   outZ.zoneFreshness = z.freshness;
   outZ.zoneMitigation = Clamp((double)z.mitigationCount / 5.0, 0.0, 1.0);
   outZ.zoneInvalidationState = (z.lifeState > 0 ? 1.0 : (z.lifeState == 0 ? 0.0 : -1.0));

   double dispWeight = Clamp(z.displacementAtr / 3.0, 0.0, 1.0);
   double freshWeight = 0.50 + 0.50 * z.freshness;
   double mitiPenalty = 1.0 - 0.12 * MathMin(z.mitigationCount, 5);
   double stateWeight = ZoneStateWeight(z);
   double confWeight = ZoneConfluenceBoost(symbol, tf, z, atr);
   double signedRel = (z.isDemand ? 1.0 : -1.0) * combinedProx * freshWeight * (0.5 + 0.5*dispWeight) * mitiPenalty * stateWeight * confWeight;
   outZ.zoneRelevance = Clamp(signedRel, -1.0, 1.0);
   outZ.zoneRetestBreakState = ZoneRetestBreakStateForTF(z, symbol, tf, atr);
}
double HighestInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = -DBL_MAX;
   for(int i=0;i<count;i++)
   {
      double v = iHigh(symbol, tf, startShift + i);
      if(v != EMPTY_VALUE && v > best) best = v;
   }
   if(best <= -DBL_MAX/2.0) best = GetCloseSafe(symbol, tf, startShift);
   return best;
}

double LowestInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = DBL_MAX;
   for(int i=0;i<count;i++)
   {
      double v = iLow(symbol, tf, startShift + i);
      if(v != EMPTY_VALUE && v < best) best = v;
   }
   if(best >= DBL_MAX/2.0) best = GetCloseSafe(symbol, tf, startShift);
   return best;
}

int HighestIndexInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = -DBL_MAX;
   int bestIdx = startShift;
   for(int i=0;i<count;i++)
   {
      int idx = startShift + i;
      double v = iHigh(symbol, tf, idx);
      if(v != EMPTY_VALUE && v > best) { best = v; bestIdx = idx; }
   }
   return bestIdx;
}

int LowestIndexInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = DBL_MAX;
   int bestIdx = startShift;
   for(int i=0;i<count;i++)
   {
      int idx = startShift + i;
      double v = iLow(symbol, tf, idx);
      if(v != EMPTY_VALUE && v < best) { best = v; bestIdx = idx; }
   }
   return bestIdx;
}
double GetCloseSafe(const string symbol,const ENUM_TIMEFRAMES tf,const int shift)
{
   double v=iClose(symbol,tf,shift);
   if(v==0.0 || v==EMPTY_VALUE)
   {
      MqlTick t;
      if(SymbolInfoTick(symbol,t)) return (t.bid+t.ask)*0.5;
   }
   return v;
}

double RollingAvgSpreadPtsTF(const string symbol,const ENUM_TIMEFRAMES tf,const int bars)
{
   int n=MathMax(bars,1);
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int copied=CopyRates(symbol,tf,1,n,rates);
   if(copied<=0) return 0.0;
   double sum=0.0;
   int cnt=0;
   for(int i=0;i<copied;i++)
   {
      if(rates[i].spread>=0)
      {
         sum += (double)rates[i].spread;
         cnt++;
      }
   }
   if(cnt<=0) return 0.0;
   return sum/(double)cnt;
}

double CopyLatestFromHandle(const int handle,const int buffer,const int shift,const double fallback=0.0)
{
   if(handle==INVALID_HANDLE) return fallback;
   double buf[];
   ArrayResize(buf,shift+1);
   ArraySetAsSeries(buf,true);
   if(CopyBuffer(handle,buffer,0,shift+1,buf) <= shift)
   {
      IndicatorRelease(handle);
      return fallback;
   }
   double v=buf[shift];
   IndicatorRelease(handle);
   if(v==EMPTY_VALUE) return fallback;
   return v;
}

double GetMAValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iMA(symbol,tf,period,0,MODE_EMA,PRICE_CLOSE);
   return CopyLatestFromHandle(h,0,shift,GetCloseSafe(symbol,tf,shift));
}

double GetRSIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iRSI(symbol,tf,period,PRICE_CLOSE);
   return CopyLatestFromHandle(h,0,shift,50.0);
}

double GetCCIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iCCI(symbol,tf,period,PRICE_TYPICAL);
   return CopyLatestFromHandle(h,0,shift,0.0);
}

double GetATRValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iATR(symbol,tf,period);
   return MathMax(CopyLatestFromHandle(h,0,shift,0.0),1e-8);
}

void GetMACDValuesTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift,double &mainVal,double &signalVal)
{
   int h=iMACD(symbol,tf,12,26,9,PRICE_CLOSE);
   mainVal   = CopyLatestFromHandle(h,0,shift,0.0);
   // handle was released above, recreate for signal
   h=iMACD(symbol,tf,12,26,9,PRICE_CLOSE);
   signalVal = CopyLatestFromHandle(h,1,shift,0.0);
}

double GetRVIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift=1)
{
   int h=iRVI(symbol,tf,10);
   return CopyLatestFromHandle(h,0,shift,0.0);
}

double GetADXValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iADX(symbol,tf,period);
   return CopyLatestFromHandle(h,0,shift,20.0);
}

double GetPlusMinusDIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iADX(symbol,tf,period);
   double plus = CopyLatestFromHandle(h,1,shift,25.0);
   h=iADX(symbol,tf,period);
   double minus = CopyLatestFromHandle(h,2,shift,25.0);
   double denom = MathMax(plus + minus, 1e-8);
   return Clamp((plus - minus) / denom, -1.0, 1.0);
}

double ReturnBarsTF(const string symbol,const ENUM_TIMEFRAMES tf,const int barsBack)
{
   double c0=GetCloseSafe(symbol,tf,1);
   double cN=GetCloseSafe(symbol,tf,1+barsBack);
   if(cN==0.0) return 0.0;
   return (c0-cN)/cN;
}

double BarRangeAtrTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift=1)
{
   double hi=iHigh(symbol,tf,shift);
   double lo=iLow(symbol,tf,shift);
   double atr=GetATRValueTF(symbol,tf,14,shift);
   return Clamp((hi-lo)/MathMax(atr,1e-8),0.0,10.0)/10.0;
}

double CloseChangeAtrTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift=1)
{
   double c0=GetCloseSafe(symbol,tf,shift);
   double c1=GetCloseSafe(symbol,tf,shift+1);
   double atr=GetATRValueTF(symbol,tf,14,shift);
   return Clamp((c0-c1)/MathMax(atr,1e-8),-5.0,5.0)/5.0;
}

double KCDistFeatureTF(const string symbol,const ENUM_TIMEFRAMES tf,const int which)
{
   double close = GetCloseSafe(symbol,tf,1);
   double ema   = GetMAValueTF(symbol,tf,20,1);
   double atr   = GetATRValueTF(symbol,tf,20,1);
   double upper = ema + 2.0*atr;
   double lower = ema - 2.0*atr;
   if(which==0) return Clamp((upper-close)/MathMax(atr,1e-8),-5.0,5.0)/5.0; // to upper
   if(which==1) return Clamp((close-lower)/MathMax(atr,1e-8),-5.0,5.0)/5.0; // to lower
   if(which==2) return Clamp((close-ema)/MathMax(atr,1e-8),-5.0,5.0)/5.0;   // to mid
   if(close>upper) return 1.0;
   if(close<lower) return -1.0;
   return 0.0;
}

double EMAFeatTF(const string symbol,const ENUM_TIMEFRAMES tf,const int mode)
{
   double close1=GetCloseSafe(symbol,tf,1);
   double close2=GetCloseSafe(symbol,tf,2);
   double ema1=GetMAValueTF(symbol,tf,20,1);
   double ema2=GetMAValueTF(symbol,tf,20,2);
   double atr=GetATRValueTF(symbol,tf,14,1);
   if(mode==0) return Clamp((close1-ema1)/MathMax(atr,1e-8),-5.0,5.0)/5.0;
   return Clamp((ema1-ema2)/MathMax(atr,1e-8),-5.0,5.0)/5.0;
}

double ATRRatioTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   double fast=GetATRValueTF(symbol,tf,7,1);
   double slow=GetATRValueTF(symbol,tf,20,1);
   return Clamp(fast/MathMax(slow,1e-8),0.0,5.0)/5.0;
}

double StdReturnsTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len,const int shiftBase=1)
{
   if(len<2) return 0.0;
   double vals[];
   ArrayResize(vals,len);
   int n=0;
   for(int i=0;i<len;i++)
   {
      double c0=GetCloseSafe(symbol,tf,shiftBase+i);
      double c1=GetCloseSafe(symbol,tf,shiftBase+i+1);
      if(c1==0.0) continue;
      vals[n++] = (c0-c1)/c1;
   }
   if(n<2) return 0.0;
   double mean=0.0;
   for(int i=0;i<n;i++) mean += vals[i];
   mean /= n;
   double var=0.0;
   for(int i=0;i<n;i++){ double d=vals[i]-mean; var += d*d; }
   var /= MathMax(n-1,1);
   return MathSqrt(MathMax(var,0.0));
}

double RealizedVolRatioTF(const string symbol,const ENUM_TIMEFRAMES tf,const int fastLen,const int slowLen)
{
   double f=StdReturnsTF(symbol,tf,fastLen,1);
   double s=StdReturnsTF(symbol,tf,slowLen,1);
   return Clamp(f/MathMax(s,1e-8),0.0,5.0)/5.0;
}

double RealizedVolLevelTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len)
{
   return Clamp(StdReturnsTF(symbol,tf,len,1)*100.0,0.0,5.0)/5.0;
}

double RealizedVolDeltaTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len)
{
   double cur=StdReturnsTF(symbol,tf,len,1);
   double prev=StdReturnsTF(symbol,tf,len,2);
   double base=MathMax(prev,1e-8);
   return Clamp((cur-prev)/base,-2.0,2.0)/2.0;
}

double RealizedVolGammaTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len)
{
   double d1=RealizedVolDeltaTF(symbol,tf,len);
   double cur=StdReturnsTF(symbol,tf,len,2);
   double prev=StdReturnsTF(symbol,tf,len,3);
   double base=MathMax(prev,1e-8);
   double d0=Clamp((cur-prev)/base,-2.0,2.0)/2.0;
   return Clamp(d1-d0,-2.0,2.0)/2.0;
}

double RangeExpansionTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shortLen,const int longLen)
{
   double sumS=0.0,sumL=0.0;
   for(int i=1;i<=shortLen;i++) sumS += iHigh(symbol,tf,i)-iLow(symbol,tf,i);
   for(int i=1;i<=longLen;i++)  sumL += iHigh(symbol,tf,i)-iLow(symbol,tf,i);
   double aS=sumS/MathMax(shortLen,1);
   double aL=sumL/MathMax(longLen,1);
   return Clamp(aS/MathMax(aL,1e-8),0.0,5.0)/5.0;
}

double HighRangeBarStreakTF(const string symbol,const ENUM_TIMEFRAMES tf,const int lookback,const double mult)
{
   double avg=0.0;
   for(int i=2;i<2+lookback;i++) avg += iHigh(symbol,tf,i)-iLow(symbol,tf,i);
   avg /= MathMax(lookback,1);
   int streak=0;
   for(int i=1;i<1+lookback;i++)
   {
      double r=iHigh(symbol,tf,i)-iLow(symbol,tf,i);
      if(r > avg*mult) streak++;
      else break;
   }
   return Clamp((double)streak/(double)MathMax(lookback,1),0.0,1.0);
}

double GetStateBudgetBase()
{
   if(EquityBudget > 1e-9)  return EquityBudget;
   if(gEAStartEquity > 1e-9) return gEAStartEquity;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > 1e-9) return eq;
   return 1.0;
}

int GetBasketDirStateFast(const string symbol,const int symIdx,const int magic)
{
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
   {
      if(gPositionsCount[symIdx] <= 0) return 0;
      if(gHasPositionTypeCache[symIdx]) return gBasketDirCache[symIdx];
   }

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      return (pt==POSITION_TYPE_BUY ? +1 : -1);
   }
   return 0;
}

double GetLastEntryPriceState(const string symbol,const int magic,const int basketDir,const double fallbackPrice)
{
   if(basketDir==0) return fallbackPrice;

   datetime lastT = 0;
   double   lastP = fallbackPrice;

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      int dir = (pt==POSITION_TYPE_BUY ? +1 : -1);
      if(dir != basketDir) continue;

      datetime t = (datetime)PositionGetInteger(POSITION_TIME);
      double   p = PositionGetDouble(POSITION_PRICE_OPEN);
      if(t >= lastT)
      {
         lastT = t;
         lastP = p;
      }
   }
   return lastP;
}

double GetBasketAgeBarsState(const int symIdx,const ENUM_TIMEFRAMES tf)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   if(gPositionsCount[symIdx] <= 0)    return 0.0;
   if(gFirstTradeTime[symIdx] <= 0)     return 0.0;

   int sec = PeriodSeconds(tf);
   if(sec <= 0) sec = PeriodSeconds(BaseTF);
   if(sec <= 0) sec = 60;

   double ageSec = (double)(TimeCurrent() - gFirstTradeTime[symIdx]);
   if(ageSec <= 0.0) return 0.0;
   return ageSec / (double)sec;
}


double RSIStateTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp((GetRSIValueTF(symbol, tf, 14, 1) - 50.0) / 30.0, -1.0, 1.0);
}

double CCIStateTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp(GetCCIValueTF(symbol, tf, 20, 1) / 200.0, -1.0, 1.0);
}

double MACDDiffStateTF(const string symbol,const ENUM_TIMEFRAMES tf,const double point)
{
   double mainV=0.0, signalV=0.0;
   GetMACDValuesTF(symbol, tf, 1, mainV, signalV);
   return Clamp((mainV - signalV) / MathMax(120.0 * point, 1e-8), -1.0, 1.0);
}

double MACDMainStateTF(const string symbol,const ENUM_TIMEFRAMES tf,const double point)
{
   double mainV=0.0, signalV=0.0;
   GetMACDValuesTF(symbol, tf, 1, mainV, signalV);
   return Clamp(mainV / MathMax(160.0 * point, 1e-8), -1.0, 1.0);
}

double RVIStateTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp(GetRVIValueTF(symbol, tf, 1), -1.0, 1.0);
}

double ADXStrengthTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp(GetADXValueTF(symbol, tf, 14, 1) / 35.0, 0.0, 1.0);
}

double VolumeRatioTF(const string symbol,const ENUM_TIMEFRAMES tf,const int lookback)
{
   long cur=iVolume(symbol, tf, 1);
   double acc=0.0;
   int used=0;
   for(int i=2;i<2+MathMax(lookback,4);i++)
   {
      long v=iVolume(symbol, tf, i);
      if(v<=0) continue;
      acc += (double)v;
      used++;
   }
   double avg=(used>0 ? acc/(double)used : (double)MathMax(cur,1));
   return Clamp(SafeDiv((double)cur, MathMax(avg,1.0), 1.0), 0.0, 3.0) / 3.0;
}

double BodyEfficiencyTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift)
{
   double openV=iOpen(symbol, tf, shift);
   double highV=iHigh(symbol, tf, shift);
   double lowV =iLow(symbol, tf, shift);
   double closeV=iClose(symbol, tf, shift);
   double range=MathMax(highV-lowV, SymbolInfoDouble(symbol,SYMBOL_POINT)*5.0);
   return Clamp(MathAbs(closeV-openV)/range, 0.0, 1.0);
}

double WickInstabilityTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift)
{
   double openV=iOpen(symbol, tf, shift);
   double highV=iHigh(symbol, tf, shift);
   double lowV =iLow(symbol, tf, shift);
   double closeV=iClose(symbol, tf, shift);
   double range=MathMax(highV-lowV, SymbolInfoDouble(symbol,SYMBOL_POINT)*5.0);
   double body=MathAbs(closeV-openV);
   return Clamp(1.0 - body/range, 0.0, 1.0);
}

void ComputeTrendFactorTF(const string symbol,
                          const ENUM_TIMEFRAMES tf,
                          const double point,
                          double &dir,
                          double &strength,
                          double &persistence,
                          double &accel,
                          double &overextension,
                          double &meanRev,
                          double &continuation,
                          double &reversalQual,
                          double &volumeAlign)
{
   double rsi=RSIStateTF(symbol,tf);
   double cci=CCIStateTF(symbol,tf);
   double macdDiff=MACDDiffStateTF(symbol,tf,point);
   double macdMain=MACDMainStateTF(symbol,tf,point);
   double rvi=RVIStateTF(symbol,tf);
   double emaDist=Clamp(EMAFeatTF(symbol,tf,0),-1.0,1.0);
   double emaSlope=Clamp(EMAFeatTF(symbol,tf,1),-1.0,1.0);
   double di=Clamp(GetPlusMinusDIValueTF(symbol,tf,14,1),-1.0,1.0);
   double adx=ADXStrengthTF(symbol,tf);
   double ret1=Clamp(ReturnBarsTF(symbol,tf,1)*100.0/2.0,-1.0,1.0);
   double ret3=Clamp(ReturnBarsTF(symbol,tf,3)*100.0/4.0,-1.0,1.0);
   double ret5=Clamp(ReturnBarsTF(symbol,tf,5)*100.0/6.0,-1.0,1.0);
   double volRatio=VolumeRatioTF(symbol,tf,10);
   double kcAbs=MathMax(MathMax(MathAbs(KCDistFeatureTF(symbol,tf,0)),MathAbs(KCDistFeatureTF(symbol,tf,1))),
                        MathMax(MathAbs(KCDistFeatureTF(symbol,tf,2)),MathAbs(KCDistFeatureTF(symbol,tf,3))));

   dir = Clamp(0.22*emaDist + 0.14*emaSlope + 0.18*macdDiff + 0.16*di + 0.10*rsi + 0.08*cci + 0.06*rvi + 0.06*ret3, -1.0, 1.0);
   strength = Clamp(0.45*MathAbs(dir) + 0.25*adx + 0.15*MathAbs(macdMain) + 0.15*MathAbs(di), 0.0, 1.0);

   double same1=(dir*ret1>0.0 ? 1.0 : 0.0);
   double same3=(dir*ret3>0.0 ? 1.0 : 0.0);
   double same5=(dir*ret5>0.0 ? 1.0 : 0.0);
   persistence = Clamp(0.35*same1 + 0.35*same3 + 0.15*same5 + 0.15*adx, 0.0, 1.0);

   double adxDelta=Clamp((GetADXValueTF(symbol,tf,14,1)-GetADXValueTF(symbol,tf,14,2))/20.0,-1.0,1.0);
   accel = Clamp(0.45*(ret1-ret3) + 0.30*adxDelta + 0.25*(macdDiff - rvi*0.5), -1.0, 1.0);

   overextension = Clamp(0.35*MathAbs(rsi) + 0.25*MathAbs(cci) + 0.25*kcAbs + 0.15*MathAbs(ret1), 0.0, 1.0);
   meanRev = Clamp(0.55*overextension + 0.20*(1.0-strength) + 0.25*MathMax(0.0, -dir*ret1), 0.0, 1.0);
   continuation = Clamp(0.42*strength + 0.20*persistence + 0.18*MathMax(0.0,accel) + 0.10*MathAbs(di) + 0.10*volRatio, 0.0, 1.0);
   reversalQual = Clamp(0.38*MathMax(0.0,-dir*ret1) + 0.24*overextension + 0.18*MathMax(0.0,-accel) + 0.10*(1.0-persistence) + 0.10*(1.0-adx), 0.0, 1.0);
   volumeAlign = Clamp(0.45*volRatio + 0.30*MathAbs(dir)*volRatio + 0.25*adx*volRatio, 0.0, 1.0);
}


double EMAReclaimStateTF(const string symbol,
                         const ENUM_TIMEFRAMES tf,
                         const int emaPeriod)
{
   if(emaPeriod<=1) return 0.0;

   int bars=iBars(symbol, tf);
   if(bars<5) return 0.0;

   int emaHandle=iMA(symbol, tf, emaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(emaHandle==INVALID_HANDLE) return 0.0;

   double emaBuf[];
   ArraySetAsSeries(emaBuf,true);
   int copied=CopyBuffer(emaHandle,0,0,4,emaBuf);
   IndicatorRelease(emaHandle);
   if(copied<3) return 0.0;

   double close1=iClose(symbol,tf,1);
   double close2=iClose(symbol,tf,2);
   double ema1=emaBuf[1];
   double ema2=emaBuf[2];
   double atr=MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);

   int side1=(close1>=ema1 ? 1 : -1);
   int side2=(close2>=ema2 ? 1 : -1);
   double distNorm=Clamp((close1-ema1)/MathMax(atr*2.0,1e-8), -1.0, 1.0);

   if(side1!=side2)
   {
      double strength=Clamp(MathAbs((close1-ema1)/MathMax(atr,1e-8)), 0.0, 1.0);
      return side1 * Clamp(0.65 + 0.35*strength, 0.0, 1.0);
   }

   return 0.35 * distNorm;
}

void ComputeMacroBiasContext(const string symbol,
                             const double point,
                             double &macroBias,
                             double &macroContinuation,
                             double &macroMaturity,
                             double &macroReclaim,
                             double &macroTransition,
                             double &lateTrendTrap)
{
   macroBias=0.0;
   macroContinuation=0.0;
   macroMaturity=0.0;
   macroReclaim=0.0;
   macroTransition=0.0;
   lateTrendTrap=0.0;

   double dirH1=0.0,strH1=0.0,persH1=0.0,accH1=0.0,overH1=0.0,mrH1=0.0,contH1=0.0,revH1=0.0,volH1=0.0;
   double dirH4=0.0,strH4=0.0,persH4=0.0,accH4=0.0,overH4=0.0,mrH4=0.0,contH4=0.0,revH4=0.0,volH4=0.0;

   ComputeTrendFactorTF(symbol, TF_LONG, point, dirH1,strH1,persH1,accH1,overH1,mrH1,contH1,revH1,volH1);
   ComputeTrendFactorTF(symbol, TF_STRUCT_EXT, point, dirH4,strH4,persH4,accH4,overH4,mrH4,contH4,revH4,volH4);

   double closeH1=GetCloseSafe(symbol, TF_LONG, 1);
   double closeH4=GetCloseSafe(symbol, TF_STRUCT_EXT, 1);

   double sDirH1=0.0,sStrH1=0.0,sContH1=0.0,sRevH1=0.0,sBreakH1=0.0,sTrapH1=0.0;
   double sDirH4=0.0,sStrH4=0.0,sContH4=0.0,sRevH4=0.0,sBreakH4=0.0,sTrapH4=0.0;
   ComputeStructureFactorTF(symbol, TF_LONG, closeH1, closeH1, closeH1, sDirH1,sStrH1,sContH1,sRevH1,sBreakH1,sTrapH1);
   ComputeStructureFactorTF(symbol, TF_STRUCT_EXT, closeH4, closeH4, closeH4, sDirH4,sStrH4,sContH4,sRevH4,sBreakH4,sTrapH4);

   double d1Dist=0.0,d1Slope=0.0,d1SideDur=0.0;
   GetD1TrendFeatures(symbol,d1Dist,d1Slope,d1SideDur);
   double d1Bias = Clamp(0.50*(d1Dist/5.0) + 0.35*(d1Slope/2.0) + 0.15*((d1Dist>=0.0 ? 1.0 : -1.0)*d1SideDur), -1.0, 1.0);
   double d1Continuation = Clamp(0.40*MathAbs(d1Bias) + 0.35*d1SideDur + 0.25*Clamp(MathAbs(d1Slope)/2.0,0.0,1.0), 0.0, 1.0);
   double d1Maturity = Clamp(0.45*Clamp(MathAbs(d1Dist)/5.0,0.0,1.0) + 0.35*d1SideDur + 0.20*(1.0- Clamp(MathAbs(d1Slope)/2.0,0.0,1.0)), 0.0, 1.0);

   double reclaimH1 = EMAReclaimStateTF(symbol, TF_LONG, 50);
   double reclaimH4 = EMAReclaimStateTF(symbol, TF_STRUCT_EXT, 50);
   double reclaimD1 = EMAReclaimStateTF(symbol, PERIOD_D1, MathMax(D1_EMA_Period,20));

   macroBias = Clamp(0.18*dirH1 + 0.12*sDirH1 + 0.20*dirH4 + 0.15*sDirH4 + 0.23*d1Bias + 0.12*Clamp(0.5*(reclaimH4+reclaimD1),-1.0,1.0), -1.0, 1.0);
   macroContinuation = Clamp(0.14*contH1 + 0.10*sContH1 + 0.20*contH4 + 0.14*sContH4 + 0.32*d1Continuation + 0.10*MathAbs(macroBias), 0.0, 1.0);
   macroMaturity = Clamp(0.14*overH1 + 0.08*MathAbs(reclaimH1) + 0.18*overH4 + 0.10*MathAbs(reclaimH4) + 0.36*d1Maturity + 0.14*MathAbs(reclaimD1), 0.0, 1.0);
   macroReclaim = Clamp(0.20*reclaimH1 + 0.35*reclaimH4 + 0.45*reclaimD1, -1.0, 1.0);
   macroTransition = Clamp(0.12*revH1 + 0.10*sRevH1 + 0.18*revH4 + 0.16*sRevH4 + 0.22*Clamp(MathAbs(macroReclaim),0.0,1.0) + 0.22*(1.0-macroContinuation), 0.0, 1.0);
   lateTrendTrap = Clamp(0.30*macroMaturity + 0.25*macroTransition + 0.20*Clamp(MathAbs(macroReclaim),0.0,1.0) + 0.15*(1.0-MathAbs(macroBias)) + 0.10*(1.0-macroContinuation), 0.0, 1.0);
}

double ComputeDirectionalMacroTrapRisk(const int action,
                                       const double macroBias,
                                       const double macroContinuation,
                                       const double macroMaturity,
                                       const double macroReclaim,
                                       const double macroTransition,
                                       const double lateTrendTrap)
{
   int dirSign=0;
   if(action==1) dirSign=1;
   else if(action==2) dirSign=-1;
   if(dirSign==0) return 0.0;

   double counterBias = MathMax(0.0, -dirSign*macroBias);
   double reclaimAgainst = MathMax(0.0, -dirSign*macroReclaim);
   double transitionAgainst = macroTransition * (0.55 + 0.45*counterBias);
   double lateMovePenalty = lateTrendTrap * (0.50 + 0.50*counterBias);

   return Clamp(0.35*counterBias +
                0.20*(macroContinuation*counterBias) +
                0.15*reclaimAgainst +
                0.15*transitionAgainst +
                0.15*lateMovePenalty, 0.0, 1.0);
}

void ComputeStructureFactorTF(const string symbol,
                              const ENUM_TIMEFRAMES tf,
                              const double price,
                              const double avgPrice,
                              const double lastEntryPrice,
                              double &dir,
                              double &strength,
                              double &continuation,
                              double &reversalQual,
                              double &breakFailureRisk,
                              double &counterTrapRisk)
{
   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
   double refPrice = GetCloseSafe(symbol, tf, 1);

   SwingPointMem swings[];
   int swingCount = 0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   ZeroMemory(lastHigh); ZeroMemory(prevHigh); ZeroMemory(lastLow); ZeroMemory(prevLow);
   bool ok = BuildSwingMemoryForTF(symbol, tf, swings, swingCount);
   bool hasRefs = false;
   if(ok)
      hasRefs = ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);
   if(!hasRefs)
      FallbackStructureRefs(symbol, tf, lastHigh, prevHigh, lastLow, prevLow);

   double tendency = SwingTendencyFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr);
   double seqStrength = SwingSequenceStrengthFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr, tendency);
   double bullBOS   = BullBOSStrengthFromMemory(refPrice, lastHigh, atr, tendency);
   double bearBOS   = BearBOSStrengthFromMemory(refPrice, lastLow, atr, tendency);
   double bullCHoCH = BullCHoCHStrengthFromMemory(refPrice, lastHigh, atr, tendency);
   double bearCHoCH = BearCHoCHStrengthFromMemory(refPrice, lastLow, atr, tendency);
   double swingNearAvg = Clamp(MathAbs(avgPrice-refPrice)/MathMax(atr*3.0,1e-8),0.0,1.0);
   double swingNearEntry = Clamp(MathAbs(lastEntryPrice-refPrice)/MathMax(atr*3.0,1e-8),0.0,1.0);

   dir = Clamp(tendency, -1.0, 1.0);
   strength = Clamp(0.60*MathAbs(seqStrength) + 0.20*MathMax(bullBOS,bearBOS) + 0.20*(1.0-0.5*(swingNearAvg+swingNearEntry)), 0.0, 1.0);
   continuation = Clamp(0.45*MathAbs(seqStrength) + 0.35*MathMax(bullBOS,bearBOS) + 0.20*MathMax(0.0,dir*seqStrength), 0.0, 1.0);
   reversalQual = Clamp(0.50*MathMax(bullCHoCH,bearCHoCH) + 0.20*(1.0-MathMax(bullBOS,bearBOS)) + 0.15*(1.0-MathAbs(seqStrength)) + 0.15*(1.0-swingNearAvg), 0.0, 1.0);
   breakFailureRisk = Clamp(0.40*MathMax(bullCHoCH,bearCHoCH) + 0.30*MathMax(0.0,1.0-MathMax(bullBOS,bearBOS)) + 0.30*swingNearEntry, 0.0, 1.0);
   counterTrapRisk = Clamp(0.55*continuation + 0.25*(1.0-reversalQual) + 0.20*MathAbs(dir), 0.0, 1.0);
}

void ComputeZoneCandleFactorTF(const string symbol,
                               const ENUM_TIMEFRAMES tf,
                               const double avgPrice,
                               const double lastEntryPrice,
                               double &zoneRelevance,
                               double &zoneRespect,
                               double &zoneInvalidRisk,
                               double &entryTiming,
                               double &trapRisk)
{
   ZoneBranchFeaturePack zf;
   double closeRef = GetCloseSafe(symbol, tf, 1);
   ComputeZoneBranchFeatures(symbol, tf, closeRef, avgPrice, lastEntryPrice, zf);

   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
   double open1 = iOpen(symbol, tf, 1);
   double high1 = iHigh(symbol, tf, 1);
   double low1  = iLow(symbol, tf, 1);
   double close1= closeRef;
   double range = MathMax(high1 - low1, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);
   double body  = MathAbs(close1 - open1);
   double upper = high1 - MathMax(open1, close1);
   double lower = MathMin(open1, close1) - low1;

   double closeLoc = Clamp((close1 - low1) / range, 0.0, 1.0);
   double rawRejectionDemand = Clamp((lower / range) * closeLoc, 0.0, 1.0);
   double rawRejectionSupply = Clamp((upper / range) * (1.0 - closeLoc), 0.0, 1.0);
   double rawRejection = (zf.zoneSideState >= 0.0 ? rawRejectionDemand : rawRejectionSupply);

   double rawAcceptanceDemand = Clamp((1.0 - closeLoc) * (body / range), 0.0, 1.0);
   double rawAcceptanceSupply = Clamp(closeLoc * (body / range), 0.0, 1.0);
   double rawAcceptance = (zf.zoneSideState >= 0.0 ? rawAcceptanceDemand : rawAcceptanceSupply);
   double indecision = Clamp(1.0 - body / range, 0.0, 1.0);
   double impulse = Clamp((body / atr), 0.0, 2.0) / 2.0;
   double activeDist = (zf.zoneSideState > 0.0 ? zf.nearestDemandDist : (zf.zoneSideState < 0.0 ? zf.nearestSupplyDist : 1.0));
   double zoneNear = 1.0 - Clamp(activeDist, 0.0, 1.0);

   zoneRelevance = Clamp(0.5 + 0.5*zf.zoneRelevance, 0.0, 1.0);
   zoneRespect = Clamp(0.45*rawRejection + 0.25*zoneNear + 0.15*(1.0-zf.zoneMitigation) + 0.15*MathAbs(zf.zoneRelevance), 0.0, 1.0);
   zoneInvalidRisk = Clamp(0.55*MathMax(0.0,-zf.zoneInvalidationState) + 0.20*rawAcceptance + 0.15*indecision + 0.10*(1.0-zoneNear), 0.0, 1.0);
   entryTiming = Clamp(0.35*zoneRespect + 0.25*impulse + 0.15*zoneNear + 0.15*(1.0-zoneInvalidRisk) + 0.10*(1.0-indecision), 0.0, 1.0);
   trapRisk = Clamp(0.35*rawAcceptance + 0.20*indecision + 0.20*zoneInvalidRisk + 0.15*(1.0-zoneRespect) + 0.10*(1.0-zoneNear), 0.0, 1.0);
}

void BuildModuleABlock(const string symbol,
                       const int symIdx,
                       const int positionsCount,
                       const int basketDirState,
                       const double avgPrice,
                       const double lastEntryPrice,
                       const double mid,
                       const double atrSlow,
                       double &out[])
{
   ArrayResize(out, ModuleAFeatureCount());
   int k=0;

   double addDepthNorm = (MaxTrades > 1 ? Clamp((double)MathMax(positionsCount - 1, 0) / (double)(MaxTrades - 1), 0.0, 1.0) : 0.0);
   double basketAgeBars = GetBasketAgeBarsState(symIdx, BaseTF);
   double basketAgeNorm = (BasketAgeNormBars > 0 ? Clamp(basketAgeBars / (double)BasketAgeNormBars, 0.0, 1.0) : 0.0);

   double distPriceToAvgAtr = 0.0;
   double distPriceToLastEntryAtr = 0.0;
   if(positionsCount > 0 && basketDirState != 0)
   {
      if(basketDirState > 0)
      {
         distPriceToAvgAtr = SafeDiv((avgPrice - mid), atrSlow, 0.0);
         distPriceToLastEntryAtr = SafeDiv((lastEntryPrice - mid), atrSlow, 0.0);
      }
      else
      {
         distPriceToAvgAtr = SafeDiv((mid - avgPrice), atrSlow, 0.0);
         distPriceToLastEntryAtr = SafeDiv((mid - lastEntryPrice), atrSlow, 0.0);
      }
   }
   distPriceToAvgAtr = Clamp(distPriceToAvgAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);
   distPriceToLastEntryAtr = Clamp(distPriceToLastEntryAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);

   double budgetBase = GetStateBudgetBase();
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double freeMarginBudgetRatio = Clamp(SafeDiv(freeMargin, budgetBase, 0.0), 0.0, StateBudgetRatioClamp) / MathMax(StateBudgetRatioClamp, 1.0);

   double worstGap = MathMax(MathAbs(distPriceToAvgAtr), MathAbs(distPriceToLastEntryAtr));
   double recoveryProgress = (positionsCount > 0 ? Clamp(1.0 - worstGap, 0.0, 1.0) : 0.0);
   double ddVelocity = Clamp(ComputeDrawdownWorseningScore(symIdx), 0.0, 1.0);
   double gridStep = GetGridStepCached(symIdx);
   double gridNorm = Clamp(SafeDiv(gridStep, MathMax(atrSlow,1e-8), 0.0), 0.0, 3.0) / 3.0;
   double addSpacingAdequacy = Clamp(0.60*gridNorm + 0.40*(1.0 - Clamp(MathAbs(distPriceToLastEntryAtr),0.0,1.0)), 0.0, 1.0);
   double rescueDependence = Clamp(0.45*addDepthNorm + 0.25*Clamp(MathAbs(distPriceToAvgAtr),0.0,1.0) + 0.15*ddVelocity + 0.15*(1.0-recoveryProgress), 0.0, 1.0);

   double recentQuality = ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate = ComputeRecentDeepBasketRate(symIdx);
   double ddCalm = ComputeRecentDrawdownCalmBonus(symIdx);

   DecisionSupportContext ds;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gDecisionSupportCache[symIdx].valid)
      CopyDecisionSupportContext(gDecisionSupportCache[symIdx], ds);
   else
      ResetDecisionSupportContext(ds);

   double oneRoundPrior = Clamp((ds.valid ? ds.oneRoundPrior : recentQuality), 0.0, 1.0);
   double deepRiskPrior = Clamp((ds.valid ? ds.addRiskPrior : recentDeepRate), 0.0, 1.0);
   double supportConfidence = Clamp((ds.valid ? ds.supportConfidence : 0.0), 0.0, 1.0);
   double archiveCaution = Clamp((ds.valid ? ds.archiveCaution : recentDeepRate), 0.0, 1.0);
   double danger = Clamp(MathMax(gPDanger[symIdx], (ds.valid ? ds.dangerProbability : 0.0)), 0.0, 1.0);
   double replayRisk = Clamp(CurrentReplayRiskBias(symIdx), 0.0, 1.0);
   double painRecurrence = Clamp((ds.valid ? ds.painRecurrenceRisk : recentDeepRate), 0.0, 1.0);
   double macroTrapPrior = Clamp((ds.valid ? ds.macroReversalTrapPrior : 0.0), 0.0, 1.0);
   double macroMicroConflict = Clamp((ds.valid ? ds.macroMicroConflict : 0.0), 0.0, 1.0);
   double painConfidence = Clamp((ds.valid ? ds.painConfidence : 0.0), 0.0, 1.0);
   double lateTrendFadePenalty = Clamp((ds.valid ? ds.lateTrendFadePenalty : 0.0), 0.0, 1.0);
   double trendPersistenceProb = Clamp((ds.valid ? ds.trendPersistenceProb : (0.40*oneRoundPrior + 0.20*supportConfidence + 0.20*recentQuality + 0.20*(1.0-danger))), 0.0, 1.0);
   double trendReversalProb    = Clamp((ds.valid ? ds.trendReversalProb    : (0.35*(1.0-painRecurrence) + 0.25*supportConfidence + 0.20*ddCalm + 0.20*(1.0-deepRiskPrior))), 0.0, 1.0);
   double spikeRiskProb        = Clamp((ds.valid ? ds.spikeRiskProb        : (0.40*danger + 0.25*archiveCaution + 0.20*replayRisk + 0.15*macroMicroConflict)), 0.0, 1.0);
   double expectedBasketDepth  = Clamp((ds.valid ? ds.expectedBasketDepth  : (0.40*deepRiskPrior + 0.20*painRecurrence + 0.15*macroTrapPrior + 0.15*macroMicroConflict + 0.10*(1.0-oneRoundPrior))), 0.0, 1.0);
   double trendContinuationQuality = Clamp((ds.valid ? ds.trendContinuationQuality : (0.30*trendPersistenceProb + 0.20*oneRoundPrior + 0.20*(1.0-deepRiskPrior) + 0.15*supportConfidence + 0.15*(1.0-macroTrapPrior))),0.0,1.0);
   double breakoutReclaimQuality  = Clamp((ds.valid ? ds.breakoutReclaimQuality  : (0.25*trendReversalProb + 0.20*supportConfidence + 0.20*(1.0-macroMicroConflict) + 0.20*oneRoundPrior + 0.15*(1.0-danger))),0.0,1.0);
   double reversalTransitionQuality = Clamp((ds.valid ? ds.reversalTransitionQuality : (0.30*macroTrapPrior + 0.20*macroMicroConflict + 0.20*lateTrendFadePenalty + 0.15*trendReversalProb + 0.15*painRecurrence)),0.0,1.0);
   double modeDominanceScore = Clamp((ds.valid ? ds.modeDominanceScore : MathMax(trendContinuationQuality,MathMax(breakoutReclaimQuality,reversalTransitionQuality)) - MathMin(MathMax(trendContinuationQuality,breakoutReclaimQuality), MathMax(MathMin(trendContinuationQuality,breakoutReclaimQuality),reversalTransitionQuality))),0.0,1.0);
   double modeConflictScore  = Clamp((ds.valid ? ds.modeConflictScore  : (1.0-modeDominanceScore + 0.25*macroMicroConflict)),0.0,1.0);

   out[k++] = (double)basketDirState;
   out[k++] = addDepthNorm;
   out[k++] = basketAgeNorm;
   out[k++] = Clamp(MathAbs(distPriceToAvgAtr), 0.0, 1.0);
   out[k++] = Clamp(MathAbs(distPriceToLastEntryAtr), 0.0, 1.0);
   out[k++] = recoveryProgress;
   out[k++] = ddVelocity;
   out[k++] = addSpacingAdequacy;
   out[k++] = rescueDependence;
   out[k++] = freeMarginBudgetRatio;
   out[k++] = oneRoundPrior;
   out[k++] = deepRiskPrior;
   out[k++] = recentQuality;
   out[k++] = ddCalm;
   out[k++] = supportConfidence;
   out[k++] = Clamp(0.50*danger + 0.30*archiveCaution + 0.20*recentDeepRate, 0.0, 1.0);
   out[k++] = replayRisk;
   out[k++] = painRecurrence;
   out[k++] = macroTrapPrior;
   out[k++] = macroMicroConflict;
   out[k++] = painConfidence;
   out[k++] = lateTrendFadePenalty;
   out[k++] = trendPersistenceProb;
   out[k++] = trendReversalProb;
   out[k++] = spikeRiskProb;
   out[k++] = expectedBasketDepth;
   out[k++] = trendContinuationQuality;
   out[k++] = breakoutReclaimQuality;
   out[k++] = reversalTransitionQuality;
   out[k++] = modeDominanceScore;
   out[k++] = modeConflictScore;
}

void BuildModuleBBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,
                       const double point,const double spreadNorm,const double spreadPts,const double hourNorm,const double dayNorm,
                       double &out[])
{
   ArrayResize(out, ModuleBFeatureCount());

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volE=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volM=0.0;
   double dirL=0.0,strL=0.0,persL=0.0,accL=0.0,overL=0.0,mrL=0.0,contL=0.0,revL=0.0,volL=0.0;

   ComputeTrendFactorTF(symbol, tfExec, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volE);
   ComputeTrendFactorTF(symbol, tfMid,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volM);
   ComputeTrendFactorTF(symbol, tfLong, point, dirL,strL,persL,accL,overL,mrL,contL,revL,volL);

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double setupConsensus = 1.0 - Clamp((MathAbs(dirE-dirM) + MathAbs(dirM-dirL) + MathAbs(dirE-dirL))/6.0, 0.0, 1.0);
   double bridgeAgreement = 1.0 - Clamp((MathAbs(dirM-dirL) + MathAbs(dirL-macroBias))/4.0, 0.0, 1.0);
   double consensus = Clamp(0.55*setupConsensus + 0.45*bridgeAgreement, 0.0, 1.0);
   double weightedDir = Clamp(0.28*dirE + 0.24*dirM + 0.18*dirL + 0.30*macroBias, -1.0, 1.0);
   double conflict = Clamp(0.60*(1.0-setupConsensus) + 0.40*(1.0-bridgeAgreement), 0.0, 1.0);
   double persistence = Clamp(0.35*persE + 0.25*persM + 0.15*persL + 0.25*macroContinuation, 0.0, 1.0);
   double accel = Clamp(0.45*accE + 0.25*accM + 0.10*accL + 0.20*macroBias, -1.0, 1.0);
   double maturity = Clamp(0.30*overE + 0.20*overM + 0.10*overL + 0.25*macroMaturity + 0.15*lateTrendTrap, 0.0, 1.0);
   double momentumStrength = Clamp(0.30*strE + 0.25*strM + 0.15*strL + 0.30*MathAbs(macroBias), 0.0, 1.0);
   double overextension = Clamp(0.40*overE + 0.20*overM + 0.10*overL + 0.20*macroMaturity + 0.10*MathAbs(macroReclaim), 0.0, 1.0);
   double meanRevPressure = Clamp(0.40*mrE + 0.25*mrM + 0.10*mrL + 0.15*macroTransition + 0.10*MathAbs(macroReclaim), 0.0, 1.0);
   double continuationPressure = Clamp(0.42*consensus*(0.50*contE + 0.30*contM + 0.20*contL) + 0.35*macroContinuation + 0.13*bridgeAgreement + 0.10*MathAbs(macroBias), 0.0, 1.0);
   double reversalConfirmation = Clamp(0.28*(0.45*revE + 0.35*revM + 0.20*revL) + 0.24*macroTransition + 0.18*MathAbs(macroReclaim) + 0.15*(1.0-macroContinuation) + 0.15*bridgeAgreement, 0.0, 1.0);
   double counterTrapRisk = Clamp(0.28*continuationPressure + 0.12*conflict + 0.10*maturity + 0.12*(1.0-reversalConfirmation) + 0.18*macroContinuation + 0.10*macroMaturity + 0.10*lateTrendTrap, 0.0, 1.0);
   double pullbackQuality = Clamp(0.30*meanRevPressure + 0.20*(1.0-counterTrapRisk) + 0.20*bridgeAgreement + 0.15*(1.0-maturity) + 0.15*MathAbs(macroReclaim), 0.0, 1.0);
   double directionalConviction = Clamp(MathAbs(weightedDir) * (0.55 + 0.25*consensus + 0.20*bridgeAgreement), 0.0, 1.0);
   double trendFreshness = Clamp(continuationPressure * (1.0 - MathMax(maturity,macroMaturity)), 0.0, 1.0);
   double volumeAlign = Clamp(0.35*volE + 0.30*volM + 0.15*volL + 0.20*MathAbs(macroBias), 0.0, 1.0);

   int k=0;
   out[k++] = dirE;
   out[k++] = dirM;
   out[k++] = dirL;
   out[k++] = consensus;
   out[k++] = conflict;
   out[k++] = persistence;
   out[k++] = accel;
   out[k++] = maturity;
   out[k++] = momentumStrength;
   out[k++] = overextension;
   out[k++] = meanRevPressure;
   out[k++] = continuationPressure;
   out[k++] = counterTrapRisk;
   out[k++] = reversalConfirmation;
   out[k++] = pullbackQuality;
   out[k++] = directionalConviction;
   out[k++] = trendFreshness;
   out[k++] = volumeAlign;
   out[k++] = macroBias;
   out[k++] = macroContinuation;
   out[k++] = macroReclaim;
   out[k++] = macroTransition;
}

void BuildModuleCBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,double &out[])
{
   ArrayResize(out, ModuleCFeatureCount());
   int symIdx=SymbolIndex(symbol);
   double volExec = Clamp(0.55*RealizedVolLevelTF(symbol, tfExec, 8) + 0.45*RealizedVolLevelTF(symbol, tfExec, 24), 0.0, 1.0);
   double volMid  = Clamp(0.55*RealizedVolLevelTF(symbol, tfMid, 8)  + 0.45*RealizedVolLevelTF(symbol, tfMid, 24), 0.0, 1.0);
   double volLong = Clamp(0.60*RealizedVolLevelTF(symbol, tfLong, 8) + 0.40*RealizedVolLevelTF(symbol, tfLong, 24), 0.0, 1.0);

   double expExec = Clamp(0.45*RealizedVolDeltaTF(symbol, tfExec, 8) + 0.25*RealizedVolGammaTF(symbol, tfExec, 8) + 0.30*RangeExpansionTF(symbol, tfExec, 5, 20), 0.0, 1.0);
   double expMid  = Clamp(0.45*RealizedVolDeltaTF(symbol, tfMid, 8)  + 0.25*RealizedVolGammaTF(symbol, tfMid, 8)  + 0.30*RangeExpansionTF(symbol, tfMid, 5, 20), 0.0, 1.0);
   double expLong = Clamp(0.55*RealizedVolDeltaTF(symbol, tfLong, 8) + 0.20*RealizedVolDeltaTF(symbol, tfLong, 20) + 0.25*Clamp(StdReturnsTF(symbol, tfLong, 20, 1)*100.0,0.0,5.0)/5.0, 0.0, 1.0);

   double shock = Clamp(0.50*expExec + 0.25*expMid + 0.15*HighRangeBarStreakTF(symbol, tfExec, 5, 1.5) + 0.10*ComputeSpreadPressureScore(symbol,symIdx), 0.0, 1.0);
   double compression = Clamp(1.0 - (0.45*volExec + 0.35*expExec + 0.20*expMid), 0.0, 1.0);

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volA=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volB=0.0;
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT); if(point<=0.0) point=0.00001;
   ComputeTrendFactorTF(symbol, tfExec, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volA);
   ComputeTrendFactorTF(symbol, tfMid,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volB);

   double noiseToTrend = Clamp((0.45*WickInstabilityTF(symbol,tfExec,1) + 0.25*WickInstabilityTF(symbol,tfMid,1) + 0.30*volExec) / MathMax(0.20,0.60*MathAbs(dirE)+0.40*strE), 0.0, 1.0);
   double spreadPressure = ComputeSpreadPressureScore(symbol,symIdx);
   double executionFriction = Clamp(0.55*spreadPressure + 0.25*noiseToTrend + 0.20*shock, 0.0, 1.0);
   double regimeBreak = Clamp(0.40*shock + 0.20*MathMax(0.0, expExec-volExec) + 0.15*MathMax(0.0, expMid-volMid) + 0.15*WickInstabilityTF(symbol,tfExec,1) + 0.10*spreadPressure, 0.0, 1.0);

   double zSignedNorm = 0.0;
   double zAbsNorm = 0.0;
   double zPause = 0.0;
   double zQuietProg = 0.0;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS && UseZScoreRiskGuard)
   {
      double denom = MathMax(ZScoreExtremeThreshold, 0.000001);
      zSignedNorm = Clamp(gZScoreLastValue[symIdx] / denom, -2.0, 2.0) / 2.0;
      zAbsNorm    = Clamp(gZScoreLastAbs[symIdx] / denom, 0.0, 2.0) / 2.0;
      zPause      = (gZScorePauseTrading[symIdx] ? 1.0 : 0.0);
      zQuietProg  = (ZScoreResumeQuietBars>0 ? Clamp((double)gZScoreQuietBars[symIdx] / (double)ZScoreResumeQuietBars, 0.0, 1.0) : 1.0);
   }

   int k=0;
   out[k++] = volExec;
   out[k++] = volMid;
   out[k++] = volLong;
   out[k++] = expExec;
   out[k++] = expMid;
   out[k++] = expLong;
   out[k++] = shock;
   out[k++] = compression;
   out[k++] = noiseToTrend;
   out[k++] = spreadPressure;
   out[k++] = executionFriction;
   out[k++] = regimeBreak;
   out[k++] = zSignedNorm;
   out[k++] = zAbsNorm;
   out[k++] = zPause;
   out[k++] = zQuietProg;
}

void BuildModuleDBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,
                       const double price,const double avgPrice,const double lastEntryPrice,double &out[])
{
   ArrayResize(out, ModuleDFeatureCount());

   double dE=0.0,sE=0.0,cE=0.0,rE=0.0,bE=0.0,tE=0.0;
   double dM=0.0,sM=0.0,cM=0.0,rM=0.0,bM=0.0,tM=0.0;
   double dL=0.0,sL=0.0,cL=0.0,rL=0.0,bL=0.0,tL=0.0;

   ComputeStructureFactorTF(symbol, tfExec, price, avgPrice, lastEntryPrice, dE,sE,cE,rE,bE,tE);
   ComputeStructureFactorTF(symbol, tfMid,  price, avgPrice, lastEntryPrice, dM,sM,cM,rM,bM,tM);
   ComputeStructureFactorTF(symbol, tfLong, price, avgPrice, lastEntryPrice, dL,sL,cL,rL,bL,tL);

   int k=0;
   out[k++] = dE; out[k++] = sE; out[k++] = cE; out[k++] = rE; out[k++] = bE; out[k++] = tE;
   out[k++] = dM; out[k++] = sM; out[k++] = cM; out[k++] = rM; out[k++] = bM; out[k++] = tM;
   out[k++] = dL; out[k++] = sL; out[k++] = cL; out[k++] = rL; out[k++] = bL; out[k++] = tL;
}

void BuildModuleEBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,
                       const double avgPrice,const double lastEntryPrice,double &out[])
{
   ArrayResize(out, ModuleEFeatureCount());

   double zrE=0.0,zsE=0.0,ziE=0.0,etE=0.0,trE=0.0;
   double zrM=0.0,zsM=0.0,ziM=0.0,etM=0.0,trM=0.0;
   double zrL=0.0,zsL=0.0,ziL=0.0,etL=0.0,trL=0.0;

   ComputeZoneCandleFactorTF(symbol, tfExec, avgPrice, lastEntryPrice, zrE,zsE,ziE,etE,trE);
   ComputeZoneCandleFactorTF(symbol, tfMid,  avgPrice, lastEntryPrice, zrM,zsM,ziM,etM,trM);
   ComputeZoneCandleFactorTF(symbol, tfLong, avgPrice, lastEntryPrice, zrL,zsL,ziL,etL,trL);

   int k=0;
   out[k++] = zrE; out[k++] = zsE; out[k++] = ziE; out[k++] = etE; out[k++] = trE;
   out[k++] = zrM; out[k++] = zsM; out[k++] = ziM; out[k++] = etM; out[k++] = trM;
   out[k++] = zrL; out[k++] = zsL; out[k++] = ziL; out[k++] = etL; out[k++] = trL;
}
void BuildModuleDFeaturesFastTF(const string symbol,
                                const ENUM_TIMEFRAMES tf,
                                const double price,
                                const double avgPrice,
                                const double lastEntryPrice,
                                const int offset,
                                double &out[])
{
   double atr = GetATRValueTF(symbol, tf, 14, 1);
   double refPrice = GetCloseSafe(symbol, tf, 1);

   SwingPointMem swings[];
   int swingCount = 0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   ZeroMemory(lastHigh); ZeroMemory(prevHigh); ZeroMemory(lastLow); ZeroMemory(prevLow);
   bool ok = BuildSwingMemoryForTF(symbol, tf, swings, swingCount);
   bool hasRefs = false;
   if(ok)
      hasRefs = ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);

   if(!hasRefs)
      FallbackStructureRefs(symbol, tf, lastHigh, prevHigh, lastLow, prevLow);

   double tendency = SwingTendencyFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr);
   double seqStrength = SwingSequenceStrengthFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr, tendency);

   double confHigh = SwingLevelConfluenceBoost(symbol, tf, lastHigh.price, true, atr);
   double confLow  = SwingLevelConfluenceBoost(symbol, tf, lastLow.price, false, atr);
   double confMean = 0.5 * (confHigh + confLow);
   tendency = Clamp(tendency * (0.85 + 0.15 * confMean), -1.0, 1.0);
   seqStrength = Clamp(seqStrength * (0.70 + 0.30 * confMean), -1.0, 1.0);

   double bullBOS   = BullBOSStrengthFromMemory(refPrice, lastHigh, atr, tendency);
   double bearBOS   = BearBOSStrengthFromMemory(refPrice, lastLow, atr, tendency);
   double bullCHoCH = 0.0;
   double bearCHoCH = 0.0;
   if(offset == 0)
   {
      bullCHoCH = BullCHoCHStrengthFromMemory(refPrice, lastHigh, atr, tendency);
      bearCHoCH = BearCHoCHStrengthFromMemory(refPrice, lastLow, atr, tendency);
   }
   double strongWeakHigh = StrongWeakHighStateFromMemory(refPrice, lastHigh, atr, tendency, seqStrength);
   double strongWeakLow  = StrongWeakLowStateFromMemory(refPrice, lastLow, atr, tendency, seqStrength);

   out[offset + 0]  = StructNormDist(lastHigh.price - refPrice, atr);
   out[offset + 1]  = StructNormDist(refPrice - lastLow.price, atr);
   out[offset + 4]  = StructNormDist(lastHigh.price - avgPrice, atr);
   out[offset + 5]  = StructNormDist(avgPrice - lastLow.price, atr);
   out[offset + 13] = tendency;
   out[offset + 16] = bullBOS;
   out[offset + 17] = bearBOS;
   if(offset == 0)
   {
      out[offset + 18] = bullCHoCH;
      out[offset + 19] = bearCHoCH;
   }
   out[offset + 20] = strongWeakHigh;
   out[offset + 21] = strongWeakLow;
}
void BuildModuleEFeaturesFastTF(const string symbol,const ENUM_TIMEFRAMES tf,const double avgPrice,const double lastEntryPrice,const int offset,double &out[])
{
   ZoneBranchFeaturePack zf;
   double closeRef = GetCloseSafe(symbol, tf, 1);
   ComputeZoneBranchFeatures(symbol, tf, closeRef, avgPrice, lastEntryPrice, zf);

   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
   double open1 = iOpen(symbol, tf, 1);
   double high1 = iHigh(symbol, tf, 1);
   double low1  = iLow(symbol, tf, 1);
   double close1= closeRef;
   double range = MathMax(high1 - low1, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);
   double body  = MathAbs(close1 - open1);
   double upper = high1 - MathMax(open1, close1);
   double lower = MathMin(open1, close1) - low1;

   double bodyAtr = Clamp(body / atr, 0.0, 5.0) / 5.0;
   double upperRatio = Clamp(upper / range, 0.0, 1.0);
   double lowerRatio = Clamp(lower / range, 0.0, 1.0);
   double closeLoc = Clamp((close1 - low1) / range, 0.0, 1.0);

   double activeDist = 1.0;
   if(zf.zoneSideState > 0.0) activeDist = zf.nearestDemandDist;
   else if(zf.zoneSideState < 0.0) activeDist = zf.nearestSupplyDist;
   double zoneNearness = 1.0 - Clamp(MathAbs(activeDist), 0.0, 1.0);
   double zoneDepthActive = 1.0 - MathAbs(zf.zoneDepthPosition);
   double swingNearness = SwingNearnessScore(symbol, tf, close1, atr);
   double structureContext = Clamp(0.50 * zoneNearness + 0.25 * zoneDepthActive + 0.15 * MathAbs(zf.zoneRelevance) + 0.10 * swingNearness, 0.0, 1.0);

   double rawRejectionDemand = Clamp((lower / range) * closeLoc, 0.0, 1.0);
   double rawRejectionSupply = Clamp((upper / range) * (1.0 - closeLoc), 0.0, 1.0);
   double rawRejection = (zf.zoneSideState >= 0.0 ? rawRejectionDemand : rawRejectionSupply);

   double rawAcceptanceDemand = Clamp((1.0 - closeLoc) * (body / range), 0.0, 1.0);
   double rawAcceptanceSupply = Clamp(closeLoc * (body / range), 0.0, 1.0);
   double rawAcceptance = (zf.zoneSideState >= 0.0 ? rawAcceptanceDemand : rawAcceptanceSupply);

   double rejection = Clamp(rawRejection * (0.55 + 0.45 * structureContext), 0.0, 1.0);
   double acceptance = Clamp(0.35 * rawAcceptance + 0.65 * rawAcceptance * structureContext, 0.0, 1.0);
   double indecisionBase = Clamp(1.0 - body / range, 0.0, 1.0);
   double indecision = Clamp(0.65 * indecisionBase + 0.35 * indecisionBase * structureContext, 0.0, 1.0);
   double impulseRaw = Clamp((body / atr) * ((close1 >= open1) ? 1.0 : -1.0), -3.0, 3.0) / 3.0;
   double impulse = Clamp(impulseRaw * (0.70 + 0.30 * structureContext), -1.0, 1.0);

   out[offset + 0] = zf.nearestDemandDist;
   out[offset + 1] = zf.nearestSupplyDist;
   out[offset + 7] = zf.zoneRelevance;
   out[offset + 8] = zf.zoneRetestBreakState;
   out[offset + 13] = rejection;
   out[offset + 14] = acceptance;
   out[offset + 15] = indecision;
   out[offset + 16] = impulse;

   if(offset == 17)
      out[offset + 15] = 0.0;
   if(offset == 34)
   {
      out[offset + 8] = 0.0;
      out[offset + 15] = 0.0;
      out[offset + 16] = 0.0;
   }
}
void BuildState(const string symbol,const int symIdx,const int positionsCount,CArrayDouble &trades,double &state[])
{
   int rawDim = 0;
   if(UseStateV2SelfAwareness) rawDim += ModuleAFeatureCount();
   if(UseIndicatorBranch)  rawDim += ModuleBFeatureCount();
   if(UseVolatilityBranch) rawDim += ModuleCFeatureCount();
   if(UseStructureBranch)  rawDim += ModuleDFeatureCount();
   if(UseZoneCandleBranch) rawDim += ModuleEFeatureCount();

   ArrayResize(state, rawDim);
   int k = 0;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   double rsiB = gRSI_Base[symIdx];
   double cciB = gCCI_Base[symIdx];
   double macdB= gMACD_BaseMain[symIdx];
   double emaB = gEMA_BaseVal[symIdx];
   double rviB = gRVI_BaseMain[symIdx];

   double normRSI = Clamp(rsiB/100.0,0.0,1.0);
   double normCCI = Clamp((cciB+500.0)/1000.0,0.0,1.0);
   double normPos = (MaxTrades>0 ? Clamp((double)positionsCount/(double)MaxTrades,0.0,1.0) : 0.0);

   double avg=mid;
   if(positionsCount>0 && trades.Total()>0)
      avg=BasketAvgPrice(trades,gTradeLots[symIdx],mid);

   double priceDiff = SafeDiv((mid-avg),(100.0*point),0.0);

   double atrRatioBase = GetAtrRatioCached_Base(symbol);

   double eq = GetEAEquity();
   double bal= gEAStartEquity;
   double eqBal = (bal>1e-9 ? eq/bal : 1.0);

   double dd=0.0;
   if(maxEquity>1e-9 && eq<maxEquity)
      dd=Clamp((maxEquity-eq)/maxEquity,0.0,1.0);

   double lastClose=iClose(symbol,BaseTF,1);
   double zoneType=0.5, zoneLoc=0.5, zoneStatus=0.5;
   GetZoneFeatures(symbol,lastClose,zoneType,zoneLoc,zoneStatus);

   double d1Dist=0.0,d1Slope=0.0,d1Side=0.0;
   GetD1TrendFeatures(symbol,d1Dist,d1Slope,d1Side);

   UpdateMultiChannelGridStats(symIdx);

   double step = GetGridStepCached(symIdx);

   double atrSlow = gATRslow_BaseVal[symIdx];
   if(atrSlow<=1e-12) atrSlow=1e-12;

   double stepNorm          = Clamp(step/atrSlow,0.0,5.0)/5.0;
   double mediumStepNorm    = stepNorm;
   double extremeStepNorm   = stepNorm;
   double activeChannelNorm = 0.0;

   double spreadPts = SafeDiv((ask-bid),point,0.0);
   double spreadNorm= Clamp(spreadPts/10.0,0.0,10.0)/10.0;

   MqlDateTime tm;
   TimeToStruct(TimeCurrent(),tm);
   double hourNorm = (double)tm.hour/23.0;
   double dayNorm  = (double)tm.day_of_week/6.0;

   double emaDirBase = Clamp((lastClose-emaB)/atrSlow,-5.0,5.0);
   double macdScaled = SafeDiv(macdB,(100.0*point),0.0);
   double rviScaled  = Clamp(rviB,-2.0,2.0)/2.0;

   int basketDirState = GetBasketDirStateFast(symbol, symIdx, gMagics[symIdx]);
   double basketState = (double)basketDirState;
   double addDepthNorm = 0.0;
   if(MaxTrades > 1)
      addDepthNorm = Clamp((double)MathMax(positionsCount - 1, 0) / (double)(MaxTrades - 1), 0.0, 1.0);

   double basketAgeBars = GetBasketAgeBarsState(symIdx, BaseTF);
   double basketAgeNorm = (BasketAgeNormBars > 0 ? Clamp(basketAgeBars / (double)BasketAgeNormBars, 0.0, 1.0) : 0.0);

   double distPriceToAvgAtr = 0.0;
   if(positionsCount > 0 && basketDirState != 0)
   {
      if(basketDirState > 0) distPriceToAvgAtr = SafeDiv((avg - mid), atrSlow, 0.0);
      else                   distPriceToAvgAtr = SafeDiv((mid - avg), atrSlow, 0.0);
      distPriceToAvgAtr = Clamp(distPriceToAvgAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);
   }

   double lastEntryPrice = GetLastEntryPriceState(symbol, gMagics[symIdx], basketDirState, avg);
   double distPriceToLastEntryAtr = 0.0;
   if(positionsCount > 0 && basketDirState != 0)
   {
      if(basketDirState > 0) distPriceToLastEntryAtr = SafeDiv((lastEntryPrice - mid), atrSlow, 0.0);
      else                   distPriceToLastEntryAtr = SafeDiv((mid - lastEntryPrice), atrSlow, 0.0);
      distPriceToLastEntryAtr = Clamp(distPriceToLastEntryAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);
   }

   double budgetBase = GetStateBudgetBase();
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double freeMarginBudgetRatio = Clamp(SafeDiv(freeMargin, budgetBase, 0.0), 0.0, StateBudgetRatioClamp) / MathMax(StateBudgetRatioClamp, 1.0);


if(UseStateV2SelfAwareness)
{
   double featsA[];
   BuildModuleABlock(symbol, symIdx, positionsCount, basketDirState, avg, lastEntryPrice, mid, atrSlow, featsA);
   for(int i=0;i<ArraySize(featsA);i++) state[k++] = featsA[i];
}

   // ===== Module B/C slow-cache blocks =====
   ENUM_TIMEFRAMES tfExec = (TF_EXEC==PERIOD_CURRENT ? BaseTF : TF_EXEC);
   ENUM_TIMEFRAMES tfMid  = TF_MID;
   ENUM_TIMEFRAMES tfLong = TF_LONG;

   datetime execBar0 = iTime(symbol, tfExec, 0);
   datetime midBar0  = iTime(symbol, tfMid, 0);
   datetime longBar0 = iTime(symbol, tfLong, 0);
   long avgQSlow = QuantizePriceByPoint(avg, point);
   long lastEntryQSlow = QuantizePriceByPoint(lastEntryPrice, point);

   if(UseIndicatorBranch)
   {
      double featsB[];
      {
         bool hit = (UseSlowFeatureCaches && CacheMatchBasic(gIndicatorCache[symIdx], symbol, execBar0, midBar0, longBar0, 0, 0, false));
         if(hit) CloneDoubleArray(gIndicatorCache[symIdx].feats, featsB);
         else
         {
            BuildModuleBBlock(symbol, tfExec, tfMid, tfLong, point, spreadNorm, spreadPts, hourNorm, dayNorm, featsB);
            if(UseSlowFeatureCaches && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gIndicatorCache[symIdx].valid=true;
               gIndicatorCache[symIdx].symbol=symbol;
               gIndicatorCache[symIdx].execBar=execBar0;
               gIndicatorCache[symIdx].midBar=midBar0;
               gIndicatorCache[symIdx].longBar=longBar0;
               CloneDoubleArray(featsB, gIndicatorCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsB);i++) state[k++] = featsB[i];
   }

   if(UseVolatilityBranch)
   {
      double featsC[];
      {
         bool hit = (UseSlowFeatureCaches && CacheMatchBasic(gVolatilityCache[symIdx], symbol, execBar0, midBar0, longBar0, 0, 0, false));
         if(hit) CloneDoubleArray(gVolatilityCache[symIdx].feats, featsC);
         else
         {
            BuildModuleCBlock(symbol, tfExec, tfMid, tfLong, featsC);
            if(UseSlowFeatureCaches && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gVolatilityCache[symIdx].valid=true;
               gVolatilityCache[symIdx].symbol=symbol;
               gVolatilityCache[symIdx].execBar=execBar0;
               gVolatilityCache[symIdx].midBar=midBar0;
               gVolatilityCache[symIdx].longBar=longBar0;
               CloneDoubleArray(featsC, gVolatilityCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsC);i++) state[k++] = featsC[i];
   }

   if(UseStructureBranch)
   {
      double featsD[];
      {
         bool hit = (UseSlowFeatureCaches && CacheStructureZoneOnExecBar &&
                     CacheMatchBasic(gStructureCache[symIdx], symbol, execBar0, midBar0, longBar0, avgQSlow, lastEntryQSlow, true));
         if(hit) CloneDoubleArray(gStructureCache[symIdx].feats, featsD);
         else
         {
            BuildModuleDBlock(symbol, tfExec, tfMid, tfLong, mid, avg, lastEntryPrice, featsD);
            if(UseSlowFeatureCaches && CacheStructureZoneOnExecBar && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gStructureCache[symIdx].valid=true;
               gStructureCache[symIdx].symbol=symbol;
               gStructureCache[symIdx].execBar=execBar0;
               gStructureCache[symIdx].midBar=midBar0;
               gStructureCache[symIdx].longBar=longBar0;
               gStructureCache[symIdx].avgQ=avgQSlow;
               gStructureCache[symIdx].lastEntryQ=lastEntryQSlow;
               CloneDoubleArray(featsD, gStructureCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsD);i++) state[k++] = featsD[i];
   }

   if(UseZoneCandleBranch)
   {
      double featsE[];
      {
         bool hit = (UseSlowFeatureCaches && CacheStructureZoneOnExecBar &&
                     CacheMatchBasic(gZoneCandleCache[symIdx], symbol, execBar0, midBar0, longBar0, avgQSlow, lastEntryQSlow, true));
         if(hit) CloneDoubleArray(gZoneCandleCache[symIdx].feats, featsE);
         else
         {
            BuildModuleEBlock(symbol, tfExec, tfMid, tfLong, avg, lastEntryPrice, featsE);
            if(UseSlowFeatureCaches && CacheStructureZoneOnExecBar && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gZoneCandleCache[symIdx].valid=true;
               gZoneCandleCache[symIdx].symbol=symbol;
               gZoneCandleCache[symIdx].execBar=execBar0;
               gZoneCandleCache[symIdx].midBar=midBar0;
               gZoneCandleCache[symIdx].longBar=longBar0;
               gZoneCandleCache[symIdx].avgQ=avgQSlow;
               gZoneCandleCache[symIdx].lastEntryQ=lastEntryQSlow;
               CloneDoubleArray(featsE, gZoneCandleCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsE);i++) state[k++] = featsE[i];
   }

   for(int i=k;i<rawDim;i++)
      state[i]=0.0;

   if(StateNormalizationL2)
   {
      InitBranchLayoutForStateDim(rawDim);
      NormalizeVecSlice(state, gBranchLayout.basketStart,     gBranchLayout.basketCount);
      NormalizeVecSlice(state, gBranchLayout.indicatorStart,  gBranchLayout.indicatorCount);
      NormalizeVecSlice(state, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount);
      NormalizeVecSlice(state, gBranchLayout.structureStart,  gBranchLayout.structureCount);
      NormalizeVecSlice(state, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount);
   }
}

bool IsExtremeState(const int symIdx,const int positionsCount)
{
   if(IsPersistentExtremeDD(symIdx))
      return true;

   return false;
}

double ScaleRewardByRegime(double reward, bool isExtreme)
{
   if(isExtreme) return reward * ExtremeRewardBoost;
   if(reward < 0.0) return reward * MildRewardScale;
   return reward;
}

bool BuildMiniStateFast(const int symIdx, double &mini[])
{
   ArrayResize(mini, MINI_DIM);

   string sym=gSymbols[symIdx];
   double point=SymbolInfoDouble(sym,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double bid=SymbolInfoDouble(sym,SYMBOL_BID);
   double ask=SymbolInfoDouble(sym,SYMBOL_ASK);

   double normPos = (MaxTrades>0 ? Clamp((double)gPositionsCount[symIdx]/(double)MaxTrades,0.0,1.0) : 0.0);
   double atrRatio = GetAtrRatioCached_Base(sym);
   double eq=GetEAEquity();
   double dd=0.0;
   if(maxEquity>1e-9 && eq<maxEquity) dd=Clamp((maxEquity-eq)/maxEquity,0.0,1.0);

   double normCCI = Clamp((gCCI_Base[symIdx]+500.0)/1000.0,0.0,1.0);

   double d1Dist=0.0,d1Slope=0.0,d1Side=0.0;
   GetD1TrendFeatures(sym,d1Dist,d1Slope,d1Side);

   double step=GetGridStepCached(symIdx);
   double atrSlow=gATRslow_BaseVal[symIdx];
   if(atrSlow<=1e-12) atrSlow=1e-12;
   double stepNorm = Clamp(step/atrSlow,0.0,5.0)/5.0;

   double spreadPts = SafeDiv((ask-bid),point,0.0);
   double spreadNorm= Clamp(spreadPts/10.0,0.0,10.0)/10.0;

   mini[0]=normPos;
   mini[1]=atrRatio;
   mini[2]=dd;
   mini[3]=normCCI;
   mini[4]=d1Dist;
   mini[5]=d1Slope;
   mini[6]=stepNorm;
   mini[7]=spreadNorm;

   NormalizeVecN(mini);
   return true;
}

void UpdateMiniStateCache(const int symIdx)
{
   if(!UseMiniStateFingerprint){ gMiniValid[symIdx]=false; return; }

   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym, BaseTF, 0);
   if(bt==0) return;
   if(bt==gMiniLastBar[symIdx] && gMiniValid[symIdx]) return;

   double mini[];
   if(!BuildMiniStateFast(symIdx, mini)){ gMiniValid[symIdx]=false; return; }

   if(gMiniValid[symIdx])
   {
      for(int k=0;k<MINI_DIM;k++) gMiniPrev[symIdx][k]=gMiniCache[symIdx][k];
      gMiniPrevValid[symIdx]=true;
   }

   for(int k=0;k<MINI_DIM;k++) gMiniCache[symIdx][k]=mini[k];
   gMiniValid[symIdx]=true;
   gMiniLastBar[symIdx]=bt;
}

bool BuildQMemoryKey(const int symIdx, const double &state[], double &key[])
{
   ArrayResize(key,0);

   int sz=ArraySize(state);
   if(sz<=0) return false;

   if(UseBranchScaffold)
      InitBranchLayoutForStateDim(sz);

   if(gBranchLayout.basketCount > 0)
   {
      int s=gBranchLayout.basketStart;
      if(gBranchLayout.basketCount>5)  Push(key, state[s+5]);   // recovery progress
      if(gBranchLayout.basketCount>10) Push(key, state[s+10]);  // one-round prior
      if(gBranchLayout.basketCount>11) Push(key, state[s+11]);  // deep-risk prior
      if(gBranchLayout.basketCount>12) Push(key, state[s+12]);  // recent quality
      if(gBranchLayout.basketCount>14) Push(key, state[s+14]);  // support confidence
      if(gBranchLayout.basketCount>15) Push(key, state[s+15]);  // danger / caution blend
      if(gBranchLayout.basketCount>17) Push(key, state[s+17]);  // pain recurrence
      if(gBranchLayout.basketCount>18) Push(key, state[s+18]);  // macro trap prior
      if(gBranchLayout.basketCount>20) Push(key, state[s+20]);  // pain confidence
      if(gBranchLayout.basketCount>22) Push(key, state[s+22]);  // persistence probability
      if(gBranchLayout.basketCount>23) Push(key, state[s+23]);  // reversal probability
      if(gBranchLayout.basketCount>24) Push(key, state[s+24]);  // spike risk probability
      if(gBranchLayout.basketCount>25) Push(key, state[s+25]);  // expected basket depth
   }

   if(gBranchLayout.indicatorCount > 0)
   {
      int s=gBranchLayout.indicatorStart;
      if(gBranchLayout.indicatorCount>3)  Push(key, state[s+3]);   // trend consensus
      if(gBranchLayout.indicatorCount>11) Push(key, state[s+11]);  // continuation pressure
      if(gBranchLayout.indicatorCount>12) Push(key, state[s+12]);  // counter-trend trap risk
      if(gBranchLayout.indicatorCount>13) Push(key, state[s+13]);  // reversal confirmation
      if(gBranchLayout.indicatorCount>15) Push(key, state[s+15]);  // directional conviction
      if(gBranchLayout.indicatorCount>18) Push(key, state[s+18]);  // macro bias
      if(gBranchLayout.indicatorCount>21) Push(key, state[s+21]);  // macro transition
   }

   if(gBranchLayout.volatilityCount > 0)
   {
      int s=gBranchLayout.volatilityStart;
      if(gBranchLayout.volatilityCount>6)  Push(key, state[s+6]);   // shock
      if(gBranchLayout.volatilityCount>9)  Push(key, state[s+9]);   // spread pressure
      if(gBranchLayout.volatilityCount>11) Push(key, state[s+11]);  // regime break
   }

   if(gBranchLayout.structureCount > 0)
   {
      int s=gBranchLayout.structureStart;
      if(gBranchLayout.structureCount>2)  Push(key, state[s+2]);    // exec continuation
      if(gBranchLayout.structureCount>3)  Push(key, state[s+3]);    // exec reversal
      if(gBranchLayout.structureCount>8)  Push(key, state[s+8]);    // mid continuation
      if(gBranchLayout.structureCount>14) Push(key, state[s+14]);   // long continuation
   }

   if(gBranchLayout.zoneCandleCount > 0)
   {
      int s=gBranchLayout.zoneCandleStart;
      if(gBranchLayout.zoneCandleCount>3)  Push(key, state[s+3]);   // exec entry timing
      if(gBranchLayout.zoneCandleCount>4)  Push(key, state[s+4]);   // exec trap risk
      if(gBranchLayout.zoneCandleCount>8)  Push(key, state[s+8]);   // mid entry timing
   }

   if(UseDangerBrain)
      Push(key, Clamp(gPDanger[symIdx],0.0,1.0));

   if(ArraySize(key)<=0) return false;

   NormalizeVecN(key);
   return true;
}


double StateBranchRelValue(const double &state[],
                           const int branchStart,
                           const int branchCount,
                           const int relIndex,
                           const double fallback=0.0)
{
   if(relIndex < 0 || branchStart < 0 || branchCount <= 0 || relIndex >= branchCount)
      return fallback;
   int idx=branchStart + relIndex;
   if(idx < 0 || idx >= ArraySize(state))
      return fallback;
   return state[idx];
}

double ComputeOptionBTrendPersistenceProb(const int symIdx,
                                          const double &state[],
                                          const double &qBase[],
                                          const DecisionSupportContext &ctx)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double consensus      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,3,0.5);
   double persistence    = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,5,0.5);
   double continuation   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);
   double reversal       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,13,0.5);
   double directionConv  = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,15,0.5);
   double volumeAlign    = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,17,0.5);
   double macroCont      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,19,0.5);
   double macroTransition= StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,21,0.5);

   double noiseToTrend   = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,8,0.5);
   double spreadPressure = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,9,0.5);
   double regimeBreak    = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double structContE    = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,2,0.5);
   double structContM    = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,8,0.5);
   double entryTimingE   = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,3,0.5);

   double recentQuality  = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,12,0.5);
   double dangerBlend    = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,15,0.5);
   double qGap=Clamp(SafeDiv(ComputeDDQNActionGap(qBase,false),0.25,0.0),0.0,1.0);

   double logit=
      -0.55
      + 1.00*consensus
      + 0.88*continuation
      + 0.72*persistence
      + 0.56*macroCont
      + 0.28*directionConv
      + 0.18*structContE
      + 0.12*structContM
      + 0.10*entryTimingE
      + 0.12*volumeAlign
      + 0.10*qGap
      + 0.08*recentQuality
      - 0.70*reversal
      - 0.48*regimeBreak
      - 0.32*noiseToTrend
      - 0.20*spreadPressure
      - 0.14*macroTransition
      - 0.10*dangerBlend;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

double ComputeOptionBTrendReversalProb(const int symIdx,
                                       const double &state[],
                                       const double &qBase[],
                                       const DecisionSupportContext &ctx)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double continuation   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);
   double reversal       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,13,0.5);
   double maturity       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,7,0.5);
   double macroReclaim   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,20,0.0);
   double macroTransition= StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,21,0.5);

   double regimeBreak    = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double structRevE     = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,3,0.5);
   double structRevM     = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,9,0.5);
   double structBreakE   = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,4,0.5);

   double zoneRespectE   = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,0,0.5);
   double zoneEntryE     = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,3,0.5);
   double zoneTrapE      = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,4,0.5);

   double macroMicroConflict = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,19,0.5);
   double lateTrendFade      = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,21,0.5);

   double qGap=Clamp(SafeDiv(ComputeDDQNActionGap(qBase,false),0.25,0.0),0.0,1.0);

   double logit=
      -0.75
      + 0.96*reversal
      + 0.82*macroTransition
      + 0.42*MathAbs(macroReclaim)
      + 0.32*maturity
      + 0.22*structRevE
      + 0.18*structRevM
      + 0.12*(1.0-structBreakE)
      + 0.10*zoneRespectE
      + 0.08*zoneEntryE
      + 0.10*macroMicroConflict
      + 0.08*lateTrendFade
      + 0.06*qGap
      - 0.70*continuation
      - 0.42*regimeBreak
      - 0.12*zoneTrapE;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

double ComputeOptionBSpikeRiskProb(const int symIdx,
                                   const double &state[],
                                   const double &qBase[],
                                   const DecisionSupportContext &ctx)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double shock          = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,6,0.5);
   double noiseToTrend   = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,8,0.5);
   double spreadPressure = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,9,0.5);
   double execFriction   = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,10,0.5);
   double regimeBreak    = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double accel          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,6,0.0);
   double volumeAlign    = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,17,0.5);

   double macroMicroConflict = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,19,0.5);

   double logit=
      -1.10
      + 1.05*shock
      + 0.92*regimeBreak
      + 0.54*spreadPressure
      + 0.50*execFriction
      + 0.44*noiseToTrend
      + 0.26*MathAbs(accel)
      + 0.20*macroMicroConflict
      + 0.18*ctx.ddEventRisk
      + 0.10*volumeAlign;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

double ComputeOptionBExpectedBasketDepth(const int symIdx,
                                         const double &state[],
                                         const double &qBase[],
                                         const DecisionSupportContext &ctx,
                                         const double persistenceProb,
                                         const double reversalProb,
                                         const double spikeRiskProb)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double recentQuality     = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,12,0.5);
   double ddCalm            = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,13,0.5);
   double dangerBlend       = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,15,0.5);
   double painRecurrence    = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,17,0.5);
   double macroTrapPrior    = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,18,0.5);
   double macroMicroConflict= StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,19,0.5);
   double lateTrendFade     = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,21,0.5);

   double trapRisk          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,12,0.5);
   double continuation      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);

   double regimeBreak       = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double oneRoundPrior=Clamp(ctx.oneRoundPrior,0.0,1.0);
   double addRiskPrior =Clamp(ctx.addRiskPrior,0.0,1.0);
   double deepPain     =Clamp(ctx.deepBasketPainPrior,0.0,1.0);

   double logit=
      -1.35
      + 1.10*addRiskPrior
      + 0.78*deepPain
      + 0.70*painRecurrence
      + 0.56*trapRisk
      + 0.44*spikeRiskProb
      + 0.38*regimeBreak
      + 0.32*macroTrapPrior
      + 0.28*macroMicroConflict
      + 0.22*continuation
      + 0.18*lateTrendFade
      + 0.16*MathMax(0.0,persistenceProb-reversalProb)
      + 0.12*dangerBlend
      - 0.62*oneRoundPrior
      - 0.38*recentQuality
      - 0.26*ddCalm;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

void ComputeStrategyModeScores(const int symIdx,
                               const double &state[],
                               const DecisionSupportContext &ctx,
                               double &trendContinuationQuality,
                               double &breakoutReclaimQuality,
                               double &reversalTransitionQuality,
                               double &modeDominanceScore,
                               double &modeConflictScore)
{
   trendContinuationQuality=0.5;
   breakoutReclaimQuality=0.5;
   reversalTransitionQuality=0.5;
   modeDominanceScore=0.0;
   modeConflictScore=0.5;

   if(ArraySize(state) <= 0) return;
   InitBranchLayoutForStateDim(ArraySize(state));

   double persistence       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,5,0.5);
   double maturity          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,7,0.5);
   double continuation      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);
   double trapRisk          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,12,0.5);
   double reversalConfirm   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,13,0.5);
   double pullbackQuality   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,14,0.5);
   double directionConv     = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,15,0.5);
   double macroBias         = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,18,0.0);
   double macroContinuation = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,19,0.5);
   double macroReclaim      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,20,0.0);
   double macroTransition   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,21,0.5);

   double regimeBreak       = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);
   double spikeRisk         = Clamp(ctx.spikeRiskProb,0.0,1.0);
   double expectedDepth     = Clamp(ctx.expectedBasketDepth,0.0,1.0);

   double structContM       = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,8,0.5);
   double structContL       = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,14,0.5);
   double structRevM        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,9,0.5);
   double structRevL        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,15,0.5);
   double breakFailM        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,10,0.5);
   double breakFailL        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,16,0.5);

   double zoneRespectM      = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,5,0.5);
   double zoneEntryM        = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,8,0.5);
   double zoneTrapM         = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,9,0.5);
   double zoneRespectL      = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,10,0.5);
   double zoneEntryL        = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,13,0.5);
   double zoneTrapL         = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,14,0.5);

   double oneRoundPrior     = Clamp(ctx.oneRoundPrior,0.0,1.0);
   double supportConf       = Clamp(ctx.supportConfidence,0.0,1.0);
   double macroConflict     = Clamp(ctx.macroMicroConflict,0.0,1.0);
   double painRecurrence    = Clamp(ctx.painRecurrenceRisk,0.0,1.0);
   double lateTrendFade     = Clamp(ctx.lateTrendFadePenalty,0.0,1.0);

   trendContinuationQuality =
      Clamp(0.16*continuation +
            0.12*pullbackQuality +
            0.12*persistence +
            0.12*macroContinuation +
            0.08*MathAbs(macroBias) +
            0.08*directionConv +
            0.08*MathMax(structContM,structContL) +
            0.06*(0.5*zoneEntryM + 0.5*zoneEntryL) +
            0.05*(0.5*zoneRespectM + 0.5*zoneRespectL) +
            0.06*(1.0-trapRisk) +
            0.04*(1.0-regimeBreak) +
            0.03*(1.0-spikeRisk) +
            0.04*(1.0-expectedDepth) +
            0.04*oneRoundPrior, 0.0, 1.0);

   double continuationRestart =
      Clamp(0.34*MathAbs(macroReclaim) +
            0.22*macroContinuation +
            0.12*continuation +
            0.10*(0.5*zoneRespectM + 0.5*zoneRespectL) +
            0.10*(0.5*zoneEntryM + 0.5*zoneEntryL) +
            0.06*MathMax(structContM,structContL) +
            0.06*(1.0-0.5*(breakFailM+breakFailL)), 0.0, 1.0);

   breakoutReclaimQuality =
      Clamp(0.28*continuationRestart +
            0.16*MathAbs(macroReclaim) +
            0.12*macroContinuation +
            0.08*directionConv +
            0.08*supportConf +
            0.06*(1.0-zoneTrapM) +
            0.05*(1.0-zoneTrapL) +
            0.06*(1.0-trapRisk) +
            0.05*(1.0-regimeBreak) +
            0.03*(1.0-spikeRisk) +
            0.03*(1.0-expectedDepth), 0.0, 1.0);

   reversalTransitionQuality =
      Clamp(0.18*macroTransition +
            0.14*maturity +
            0.14*reversalConfirm +
            0.12*MathAbs(macroReclaim) +
            0.10*macroConflict +
            0.08*MathMax(structRevM,structRevL) +
            0.06*(0.5*breakFailM + 0.5*breakFailL) +
            0.05*regimeBreak +
            0.04*spikeRisk +
            0.04*painRecurrence +
            0.03*lateTrendFade +
            0.02*(1.0-macroContinuation), 0.0, 1.0);

   double a=trendContinuationQuality, b=breakoutReclaimQuality, c=reversalTransitionQuality;
   double best=MathMax(a, MathMax(b,c));
   double second=MathMin(MathMax(a,b), MathMax(MathMin(a,b),c));
   modeDominanceScore=Clamp(best - second, 0.0, 1.0);
   modeConflictScore=Clamp(1.0 - modeDominanceScore + 0.25*macroConflict, 0.0, 1.0);
}



bool QueryQMemory(const int symIdx,
                  const int regime,
                  const double &key[],
                  double &qOut[],
                  double &confOut)
{
   ArrayResize(qOut, ActionCount);
   for(int a=0;a<ActionCount;a++) qOut[a]=0.0;
   confOut=0.0;

   if(!UseQMemory) return false;
   if(ArraySize(key)<=0) return false;

   double sumW=0.0;
   double bestSim=-1e9;
   int bestIdx=-1;

   int n=ArraySize(gQMem);
   for(int i=0;i<n;i++)
   {
      if(gQMem[i].regime != regime) continue;
      if(ArraySize(gQMem[i].stateKey)!=ArraySize(key)) continue;
      if(ArraySize(gQMem[i].qVals)!=ActionCount) continue;

      double sim = QMemSimilarity(key, gQMem[i].stateKey);
      if(sim < 0.0) continue;

      double w = sim * gQMem[i].conf * MathMax(0.0, gQMem[i].score) * QMemAgeFactor(gQMem[i]);
      if(w<=1e-12) continue;

      for(int a=0;a<ActionCount;a++)
         qOut[a] += w * gQMem[i].qVals[a];

      sumW += w;

      if(sim>bestSim)
      {
         bestSim=sim;
         bestIdx=i;
      }
   }

   if(sumW<=1e-12 || bestIdx<0) return false;

   for(int a=0;a<ActionCount;a++)
      qOut[a] /= sumW;

   confOut = Clamp(bestSim,0.0,1.0);

   gQMem[bestIdx].usedCount++;
   gQMem[bestIdx].lastUsed=TimeCurrent();

   return (confOut >= QMemMinConfidence);
}

datetime FastTrainingQMemoryDecisionBarTime(const string symbol)
{
   datetime barTime = iTime(symbol, BaseTF, 1);
   if(barTime<=0) barTime = iTime(symbol, BaseTF, 0);
   if(barTime<=0) barTime = TimeCurrent();
   return barTime;
}

bool ShouldUseSparseQMemoryRetrieval(const int symIdx)
{
   if(!FastTrainingMode) return false;
   if(!FastTrainingSparseQMemory) return false;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return false;
   return true;
}

bool GetQMemoryForDecision(const int symIdx,
                           const int regime,
                           const string symbol,
                           const double &stateKey[],
                           double &qOut[],
                           double &confOut)
{
   ArrayResize(qOut, ActionCount);
   for(int a=0;a<ActionCount;a++) qOut[a]=0.0;
   confOut=0.0;

   if(!UseQMemory) return false;
   if(ArraySize(stateKey)<=0) return false;

   if(!ShouldUseSparseQMemoryRetrieval(symIdx))
      return QueryQMemory(symIdx, regime, stateKey, qOut, confOut);

   datetime barTime = FastTrainingQMemoryDecisionBarTime(symbol);
   if(barTime<=0) barTime = TimeCurrent();

   bool barAdvanced = (gFastQMemLastDecisionBarTime[symIdx] != barTime);
   if(barAdvanced)
   {
      gFastQMemLastDecisionBarTime[symIdx] = barTime;
      gFastQMemDecisionBarCounter[symIdx]++;
   }

   int refreshBars = MathMax(1, FastTrainingQMemoryRefreshDecisionBars);
   int refreshMins = MathMax(1, FastTrainingQMemoryRefreshMinutes);

   bool needRefresh = (!gFastQMemCachedValid[symIdx] || gFastQMemCachedRegime[symIdx] != regime);
   if(!needRefresh && gFastQMemDecisionBarCounter[symIdx] >= refreshBars)
      needRefresh = true;
   if(!needRefresh && (TimeCurrent() - gFastQMemLastRefreshTime[symIdx]) >= (refreshMins * 60))
      needRefresh = true;

   if(needRefresh)
   {
      double qTmp[];
      double confTmp = 0.0;
      bool ok = QueryQMemory(symIdx, regime, stateKey, qTmp, confTmp);

      gFastQMemCachedValid[symIdx]   = ok;
      gFastQMemCachedRegime[symIdx]  = regime;
      gFastQMemCachedConf[symIdx]    = confTmp;
      gFastQMemLastRefreshTime[symIdx] = TimeCurrent();
      gFastQMemDecisionBarCounter[symIdx] = 0;

      for(int a=0;a<3;a++)
         gFastQMemCachedQ[symIdx][a] = (a < ArraySize(qTmp) ? qTmp[a] : 0.0);
   }

   if(!gFastQMemCachedValid[symIdx] || gFastQMemCachedRegime[symIdx] != regime)
      return false;

   for(int a=0;a<ActionCount && a<3;a++)
      qOut[a] = gFastQMemCachedQ[symIdx][a];
   confOut = gFastQMemCachedConf[symIdx];
   return (confOut >= QMemMinConfidence);
}

void UpdateQMemoryResolved(const int symIdx,
                           const int regime,
                           const double &state[],
                           const int action,
                           const double reward)
{
   if(!UseQMemory) return;
   if(action<0 || action>=ActionCount) return;

   double key[];
   if(!BuildQMemoryKey(symIdx,state,key)) return;

   double st[];
   ArrayResize(st,ArraySize(state));
   for(int i=0;i<ArraySize(state);i++) st[i]=state[i];

   double qBase[];
   ArrayResize(qBase,ActionCount);
   DQNForward(symIdx,regime,st,qBase);

   double qTarget[];
   ArrayResize(qTarget,ActionCount);
   for(int a=0;a<ActionCount;a++) qTarget[a]=qBase[a];

   double targetReward = reward;
   if(targetReward < 0.0 && QMemUseNegativeUpdates)
      targetReward *= QMemNegativePenaltyScale;
   else if(targetReward > 0.0)
      targetReward *= QMemPositiveBoostScale;

   qTarget[action] = targetReward;

   double bestSim=-1e9;
   int best=-1;

   int n=ArraySize(gQMem);
   for(int i=0;i<n;i++)
   {
      if(gQMem[i].regime!=regime) continue;
      if(ArraySize(gQMem[i].stateKey)!=ArraySize(key)) continue;
      if(ArraySize(gQMem[i].qVals)!=ActionCount) continue;

      double sim=QMemSimilarity(key,gQMem[i].stateKey);
      if(sim>bestSim){ bestSim=sim; best=i; }
   }

   if(best>=0 && bestSim>=QMemMergeSim)
   {
      for(int k=0;k<ArraySize(key);k++)
         gQMem[best].stateKey[k] = (1.0-QMemFeatureEMA)*gQMem[best].stateKey[k] + QMemFeatureEMA*key[k];
      NormalizeVecN(gQMem[best].stateKey);

      for(int a=0;a<ActionCount;a++)
         gQMem[best].qVals[a] = (1.0-QMemFeatureEMA)*gQMem[best].qVals[a] + QMemFeatureEMA*qTarget[a];

      if(reward>=0.0)
      {
         gQMem[best].conf  = Clamp(gQMem[best].conf + 0.05, 0.0, 1.0);
         gQMem[best].score = MathMax(0.0, gQMem[best].score + 0.20);
      }
      else
      {
         gQMem[best].conf  = Clamp(gQMem[best].conf - 0.03, 0.0, 1.0);
         gQMem[best].score = MathMax(0.0, gQMem[best].score - 0.10);
      }

      gQMem[best].usedCount++;
      gQMem[best].lastUsed=TimeCurrent();
   }
   else
   {
      QMemEntry e;
      e.regime=regime;
      e.created=TimeCurrent();
      e.lastUsed=TimeCurrent();
      e.usedCount=0;

      ArrayResize(e.stateKey,ArraySize(key));
      for(int k=0;k<ArraySize(key);k++) e.stateKey[k]=key[k];

      ArrayResize(e.qVals,ActionCount);
      for(int a=0;a<ActionCount;a++) e.qVals[a]=qTarget[a];

      e.conf  = (reward>=0.0 ? 0.35 : 0.20);
      e.score = (reward>=0.0 ? 1.0 : 0.4);

      int m=ArraySize(gQMem);
      ArrayResize(gQMem,m+1);
      gQMem[m]=e;
   }

   PruneQMemoryIfNeeded();
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
      gDecisionSupportCache[symIdx].valid=false;
}



double CurrentReplayDDPct()
{
   double eq = GetEAEquity();
   double peak = MathMax(maxEquity, 1e-8);
   return MathMax(0.0, (peak - eq) / peak);
}

double GetCurrentMidPriceForSymbol(const string symbol)
{
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   if(bid<=0.0 && ask<=0.0) return 0.0;
   if(bid<=0.0) return ask;
   if(ask<=0.0) return bid;
   return 0.5*(bid+ask);
}

double GetLastEntryPriceFromCache(const int symIdx, const double fallbackPrice)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return fallbackPrice;
   int n = gTrades[symIdx].Total();
   if(n<=0) return fallbackPrice;
   return gTrades[symIdx].At(n-1);
}

int ClassifyReplayBasketState(const int symIdx)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return 0;
   int pc = gPositionsCount[symIdx];
   if(pc<=0) return 0;
   if(pc==1) return 1;
   if(pc<=3) return 2;
   return 3;
}

int ClassifyReplayAddDepth(const int symIdx)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return 0;
   int adds = MathMax(gPositionsCount[symIdx]-1, 0);
   if(adds<=0) return 0;
   if(adds<=2) return 1;
   if(adds<=DeepBasketAddThreshold) return 2;
   return 3;
}

int ClassifyReplayDanger(const int symIdx, const double reward)
{
   double dd = GetSymbolFloatingDDPct(symIdx);
   if(dd >= DangerReplayDDThresholdPct || reward <= -3.0) return 2;
   if(dd >= 0.5*DangerReplayDDThresholdPct || reward <= -1.0) return 1;
   return 0;
}

int ClassifyReplayLiquidity(const string symbol)
{
   double spr = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   double avgSpr = RollingAvgSpreadPtsTF(symbol, TF_EXEC, 64);
   if(avgSpr <= 0.0) avgSpr = MathMax(spr, 1.0);
   double ratio = spr / MathMax(avgSpr, 1e-8);
   if(ratio >= 1.75) return 2;
   if(ratio >= 1.20) return 1;
   return 0;
}


long EnsureActiveBasketEpisode(const int symIdx,
                              const int basketDir,
                              const int positionsAtOpen,
                              const datetime entryTime)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0;

   if(!gActiveBasketEpisodes[symIdx].active)
   {
      gActiveBasketEpisodes[symIdx].active=true;
      gActiveBasketEpisodes[symIdx].episodeId=gNextEpisodeId++;
      gActiveBasketEpisodes[symIdx].symIdx=symIdx;
      gActiveBasketEpisodes[symIdx].basketDir=basketDir;
      gActiveBasketEpisodes[symIdx].startTime=(entryTime>0 ? entryTime : TimeCurrent());
      gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
      gActiveBasketEpisodes[symIdx].addCount=MathMax(positionsAtOpen-1,0);
      gActiveBasketEpisodes[symIdx].maxPositions=MathMax(positionsAtOpen,1);
      gActiveBasketEpisodes[symIdx].maxDD=GetSymbolFloatingDDPct(symIdx);
      gActiveBasketEpisodes[symIdx].rewardAccum=0.0;
      ArrayResize(gActiveBasketEpisodes[symIdx].replayItemIndexes,0);
   }
   else
   {
      gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
      gActiveBasketEpisodes[symIdx].basketDir=basketDir;
      gActiveBasketEpisodes[symIdx].addCount=MathMax(gActiveBasketEpisodes[symIdx].addCount, MathMax(positionsAtOpen-1,0));
      gActiveBasketEpisodes[symIdx].maxPositions=MathMax(gActiveBasketEpisodes[symIdx].maxPositions, MathMax(positionsAtOpen,1));
      gActiveBasketEpisodes[symIdx].maxDD=MathMax(gActiveBasketEpisodes[symIdx].maxDD, GetSymbolFloatingDDPct(symIdx));
   }
   return gActiveBasketEpisodes[symIdx].episodeId;
}

long GetCurrentEpisodeIdForSymbol(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0;
   if(!gActiveBasketEpisodes[symIdx].active) return 0;
   return gActiveBasketEpisodes[symIdx].episodeId;
}

void AppendReplayIndexToActiveEpisode(const int symIdx,const int replayIdx,const double reward)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(!gActiveBasketEpisodes[symIdx].active) return;
   int n=ArraySize(gActiveBasketEpisodes[symIdx].replayItemIndexes);
   ArrayResize(gActiveBasketEpisodes[symIdx].replayItemIndexes,n+1);
   gActiveBasketEpisodes[symIdx].replayItemIndexes[n]=replayIdx;
   gActiveBasketEpisodes[symIdx].rewardAccum += reward;
   gActiveBasketEpisodes[symIdx].maxDD=MathMax(gActiveBasketEpisodes[symIdx].maxDD, GetSymbolFloatingDDPct(symIdx));
   gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
}

int EstimateSessionType(const datetime t)
{
   MqlDateTime dt;
   TimeToStruct((t>0 ? t : TimeCurrent()), dt);
   int h=dt.hour;
   if(h<8) return 0;
   if(h<16) return 1;
   return 2;
}

bool GetDecisionContextCached(const int symIdx,
                              const int regimeHint,
                              int &regimeType,
                              int &patternType,
                              int &liquidityType,
                              int &sessionType,
                              double &atrRatio)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return false;

   string symbol = gSymbols[symIdx];
   datetime barTime = iTime(symbol, BaseTF, 1);
   datetime nowT = TimeCurrent();
   int hourBucket = (int)(nowT / 3600);

   bool refresh = (!gDecisionCtxValid[symIdx]);
   if(!refresh && gDecisionCtxBarTime[symIdx] != barTime) refresh = true;
   if(!refresh && gDecisionCtxHourBucket[symIdx] != hourBucket) refresh = true;

   if(refresh)
   {
      atrRatio = GetAtrRatioCached_Base(symbol);
      gDecisionCtxAtrRatio[symIdx] = atrRatio;
      gDecisionCtxRegimeType[symIdx] = RegimeIndexFromRatio(atrRatio);
      gDecisionCtxPatternType[symIdx] = ClassifyReplayStructureContext(symbol);
      gDecisionCtxLiquidityType[symIdx] = ClassifyReplayLiquidity(symbol);
      gDecisionCtxSessionType[symIdx] = EstimateSessionType(nowT);
      gDecisionCtxBarTime[symIdx] = barTime;
      gDecisionCtxHourBucket[symIdx] = hourBucket;
      gDecisionCtxValid[symIdx] = true;
   }

   atrRatio = gDecisionCtxAtrRatio[symIdx];
   regimeType = gDecisionCtxRegimeType[symIdx];
   if(regimeHint>=0 && regimeHint<REGIME_COUNT)
      regimeType = regimeHint;
   patternType = gDecisionCtxPatternType[symIdx];
   liquidityType = gDecisionCtxLiquidityType[symIdx];
   sessionType = gDecisionCtxSessionType[symIdx];
   return true;
}


BarSpanRef MakeBarSpanRef(const string symbol,const ENUM_TIMEFRAMES tf,const datetime startT,const datetime endT)
{
   BarSpanRef r;
   r.tf=tf;
   r.startTime=startT;
   r.endTime=endT;
   r.startIndex=iBarShift(symbol, tf, startT, false);
   r.endIndex=iBarShift(symbol, tf, endT, false);
   return r;
}

void PruneArchiveMemoriesIfNeeded()
{
   while(ArraySize(gEpisodeMemory) > MaxEpisodeMemory) ArrayRemove(gEpisodeMemory,0,1);
   while(ArraySize(gPatternMemory) > MaxPatternMemory) ArrayRemove(gPatternMemory,0,1);
   while(ArraySize(gRegimeEventMemory) > MaxRegimeEventMemory) ArrayRemove(gRegimeEventMemory,0,1);
}

long NextPatternMemoryId()
{
   int n=ArraySize(gPatternMemory);
   return (n>0 ? (gPatternMemory[n-1].patternId + 1) : 1);
}

long NextRegimeEventMemoryId()
{
   int n=ArraySize(gRegimeEventMemory);
   return (n>0 ? (gRegimeEventMemory[n-1].regimeId + 1) : 1);
}

int PatternStrengthClassFromEpisode(const EpisodeMemory &ep)
{
   if(ep.rewardEfficiency > 0.0 && ep.maxDrawdownPct <= RewardV2CleanCycleMaxDD && ep.addCount <= 1)
      return 2;
   if(ep.rewardEfficiency < 0.0 || ep.maxDrawdownPct >= DangerReplayDDThresholdPct || ep.addCount >= DeepBasketAddThreshold)
      return 0;
   return 1;
}

void AppendArchiveRecordsFromBasketSequence(const int symIdx,const BasketSequenceRef &seq)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   string symbol = gSymbols[symIdx];
   datetime startT = (seq.startTime>0 ? seq.startTime : TimeCurrent());
   datetime endT   = (seq.endTime>=startT ? seq.endTime : TimeCurrent());

   EpisodeMemory ep;
   ep.episodeId         = seq.episodeId;
   ep.symbol            = symbol;
   ep.symIdx            = symIdx;
   ep.startTime         = startT;
   ep.endTime           = endT;
   ep.basketDir         = seq.basketDir;
   ep.openPositionsMax  = MathMax(seq.maxPositions,1);
   ep.addCount          = MathMax(seq.addCount,0);
   ep.actionsCount      = MathMax(ArraySize(seq.replayItemIndexes), ep.addCount + 1);
   ep.entryPriceFirst   = 0.0;
   ep.avgEntryAtWorst   = 0.0;
   ep.closePriceFinal   = 0.0;
   ep.pnlFinal          = 0.0;
   ep.rewardTotal       = seq.finalReward;
   double durationMin   = MathMax((double)(endT - startT) / 60.0, 1.0);
   ep.rewardEfficiency  = seq.finalReward / durationMin;
   ep.maxDrawdownPct    = MathMax(seq.maxDD, 0.0);
   ep.maxDangerScore    = CurrentReplayRiskBias(symIdx);
   ep.maxMarginStress   = CurrentMarginStressRatio();
   ep.oneRoundTrade     = (ep.openPositionsMax <= 1 ? 1 : 0);

   bool riskyProfit = (ep.rewardTotal > 0.0 && (ep.addCount > 0 || ep.openPositionsMax > 2 || ep.maxDrawdownPct > RewardV2CleanCycleMaxDD));
   ep.forcedStopLikeEvent = ((ep.rewardTotal <= 0.0 && ep.maxDrawdownPct >= DangerReplayDDThresholdPct) ||
                             ep.openPositionsMax >= MathMax(DeepBasketAddThreshold + 1, 4) ? 1 : 0);
   ep.inefficientRecovery = ((ep.addCount > 0) &&
                             (ep.rewardTotal <= 0.0 || ep.maxDrawdownPct > RewardV2CleanCycleMaxDD ||
                              ep.openPositionsMax > 2 || riskyProfit) ? 1 : 0);

   ep.sessionType       = EstimateSessionType(endT);
   ep.liquidityType     = ClassifyReplayLiquidity(symbol);
   ep.regimeType        = RegimeIndexFromRatio(GetAtrRatioCached_Base(symbol));
   ep.patternType       = ClassifyReplayStructureContext(symbol);

   ep.execSpan          = MakeBarSpanRef(symbol, TF_EXEC, startT, endT);
   ep.midSpan           = MakeBarSpanRef(symbol, TF_MID, startT, endT);
   ep.longSpan          = MakeBarSpanRef(symbol, TF_LONG, startT, endT);
   ep.structExtSpan     = MakeBarSpanRef(symbol, TF_STRUCT_EXT, startT, endT);

   int ne=ArraySize(gEpisodeMemory);
   ArrayResize(gEpisodeMemory, ne+1);
   gEpisodeMemory[ne]=ep;

   PatternMemory pm;
   pm.patternId            = NextPatternMemoryId();
   pm.symbol               = symbol;
   pm.symIdx               = symIdx;
   pm.patternType          = ep.patternType;
   pm.strengthClass        = PatternStrengthClassFromEpisode(ep);
   pm.volatilityClass      = ep.regimeType;
   pm.liquidityClass       = ep.liquidityType;
   pm.startTime            = startT;
   pm.endTime              = endT;
   pm.rewardEfficiencyMean = ep.rewardEfficiency;
   pm.ddMean               = ep.maxDrawdownPct;
   pm.addMean              = (double)ep.addCount;
   pm.execSpan             = ep.execSpan;
   pm.midSpan              = ep.midSpan;
   pm.longSpan             = ep.longSpan;
   ArrayResize(pm.linkedEpisodeIds,1);
   pm.linkedEpisodeIds[0]  = ep.episodeId;

   int np=ArraySize(gPatternMemory);
   ArrayResize(gPatternMemory, np+1);
   gPatternMemory[np]=pm;

   RegimeEventMemory rm;
   rm.regimeId            = NextRegimeEventMemoryId();
   rm.symbol              = symbol;
   rm.symIdx              = symIdx;
   rm.regimeType          = ep.regimeType;
   rm.eventType           = ((ep.forcedStopLikeEvent!=0 || ep.inefficientRecovery!=0 || riskyProfit) ? 1 : 0);
   rm.startTime           = startT;
   rm.endTime             = endT;
   rm.avgVol              = GetAtrRatioCached_Base(symbol);
   rm.avgSpread           = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   rm.avgADX              = GetADXValueTF(symbol, TF_EXEC, 14, 1);
   rm.avgRewardEfficiency = ep.rewardEfficiency;
   rm.execSpan            = ep.execSpan;
   rm.midSpan             = ep.midSpan;
   rm.longSpan            = ep.longSpan;
   ArrayResize(rm.linkedEpisodeIds,1);
   rm.linkedEpisodeIds[0] = ep.episodeId;
   ArrayResize(rm.linkedPatternIds,1);
   rm.linkedPatternIds[0] = pm.patternId;

   int nr=ArraySize(gRegimeEventMemory);
   ArrayResize(gRegimeEventMemory, nr+1);
   gRegimeEventMemory[nr]=rm;

   PruneArchiveMemoriesIfNeeded();
}


void FinalizeActiveBasketEpisode(const int symIdx,const double finalReward)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(!gActiveBasketEpisodes[symIdx].active) return;

   BasketSequenceRef seq;
   seq.episodeId   = gActiveBasketEpisodes[symIdx].episodeId;
   seq.symIdx      = symIdx;
   seq.startTime   = gActiveBasketEpisodes[symIdx].startTime;
   seq.endTime     = gActiveBasketEpisodes[symIdx].lastTime;
   seq.basketDir   = gActiveBasketEpisodes[symIdx].basketDir;
   seq.addCount    = gActiveBasketEpisodes[symIdx].addCount;
   seq.maxPositions= gActiveBasketEpisodes[symIdx].maxPositions;
   seq.maxDD       = gActiveBasketEpisodes[symIdx].maxDD;
   seq.finalReward = gActiveBasketEpisodes[symIdx].rewardAccum + finalReward;

   ArrayResize(seq.replayItemIndexes, ArraySize(gActiveBasketEpisodes[symIdx].replayItemIndexes));
   for(int i=0;i<ArraySize(seq.replayItemIndexes);i++)
      seq.replayItemIndexes[i]=gActiveBasketEpisodes[symIdx].replayItemIndexes[i];

   if(DeepBasketSequenceQualifies(seq))
   {
      int n=ArraySize(gDeepBasketReplayBank.sequences);
      ArrayResize(gDeepBasketReplayBank.sequences,n+1);
      gDeepBasketReplayBank.sequences[n]=seq;
      if(ArraySize(gDeepBasketReplayBank.sequences) > MathMax(DeepBasketReplayCapacity/4, 256))
         ArrayRemove(gDeepBasketReplayBank.sequences,0,1);
      gDeepBasketReplayBank.seqCount = ArraySize(gDeepBasketReplayBank.sequences);
   }

   AppendArchiveRecordsFromBasketSequence(symIdx, seq);

   gActiveBasketEpisodes[symIdx].active=false;
   gActiveBasketEpisodes[symIdx].episodeId=0;
   ArrayResize(gActiveBasketEpisodes[symIdx].replayItemIndexes,0);
}

void EnsureActiveEfficientPeriod(const int symIdx,const datetime t)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(gActiveEfficientPeriods[symIdx].active)
      return;

   gActiveEfficientPeriods[symIdx].active=true;
   gActiveEfficientPeriods[symIdx].periodId=gNextPeriodId++;
   gActiveEfficientPeriods[symIdx].symIdx=symIdx;
   gActiveEfficientPeriods[symIdx].startTime=t;
   gActiveEfficientPeriods[symIdx].lastTime=t;
   gActiveEfficientPeriods[symIdx].rewardTotal=0.0;
   gActiveEfficientPeriods[symIdx].ddMax=GetSymbolFloatingDDPct(symIdx);
   gActiveEfficientPeriods[symIdx].oneRoundCount=0;
   gActiveEfficientPeriods[symIdx].addCountTotal=0;
   gActiveEfficientPeriods[symIdx].tradeCount=0;
   gActiveEfficientPeriods[symIdx].lastLiquidityClass=-1;
   gActiveEfficientPeriods[symIdx].lastRegime=-1;
   gActiveEfficientPeriods[symIdx].lastStructureClass=-1;
   gActiveEfficientPeriods[symIdx].contextBreakCount=0;
   ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,0);
}

void FinalizeActiveEfficientPeriod(const int symIdx,const int sessionType,const int liquidityClass,const int regimeType,const int patternType)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(!gActiveEfficientPeriods[symIdx].active) return;
   if(gActiveEfficientPeriods[symIdx].tradeCount < EfficientPeriodMinTrades)
   {
      gActiveEfficientPeriods[symIdx].active=false;
      gActiveEfficientPeriods[symIdx].periodId=0;
      ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,0);
      return;
   }

   EfficientPeriodRef pr;
   pr.periodId = gActiveEfficientPeriods[symIdx].periodId;
   pr.symIdx   = symIdx;
   pr.startTime= gActiveEfficientPeriods[symIdx].startTime;
   pr.endTime  = gActiveEfficientPeriods[symIdx].lastTime;
   pr.rewardTotal = gActiveEfficientPeriods[symIdx].rewardTotal;
   double durationMin = MathMax((double)(pr.endTime - pr.startTime)/60.0, 1.0);
   pr.rewardEfficiency = pr.rewardTotal / durationMin;
   pr.ddMax = gActiveEfficientPeriods[symIdx].ddMax;
   pr.oneRoundCount = gActiveEfficientPeriods[symIdx].oneRoundCount;
   pr.addCountTotal = gActiveEfficientPeriods[symIdx].addCountTotal;
   pr.sessionType = sessionType;
   pr.liquidityType = liquidityClass;
   pr.regimeType = regimeType;
   pr.patternType = patternType;
   ArrayResize(pr.replayItemIndexes, ArraySize(gActiveEfficientPeriods[symIdx].replayItemIndexes));
   for(int i=0;i<ArraySize(pr.replayItemIndexes);i++)
      pr.replayItemIndexes[i]=gActiveEfficientPeriods[symIdx].replayItemIndexes[i];

   if(EfficientPeriodQualifies(pr))
   {
      int n=ArraySize(gEfficientReplayBank.periods);
      ArrayResize(gEfficientReplayBank.periods,n+1);
      gEfficientReplayBank.periods[n]=pr;
      if(ArraySize(gEfficientReplayBank.periods) > MathMax(EfficientReplayCapacity/4, 256))
         ArrayRemove(gEfficientReplayBank.periods,0,1);
      gEfficientReplayBank.periodCount = ArraySize(gEfficientReplayBank.periods);
   }

   gActiveEfficientPeriods[symIdx].active=false;
   gActiveEfficientPeriods[symIdx].periodId=0;
   ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,0);
}

void UpdateActiveEfficientPeriodWithItem(const ReplayItem &item,const int replayIdx)
{
   int symIdx=item.symIdx;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   datetime nowT = (item.eventTime>0 ? item.eventTime : TimeCurrent());

   if(gActiveEfficientPeriods[symIdx].active)
   {
      int elapsedMin = (int)((nowT - gActiveEfficientPeriods[symIdx].startTime)/60);
      bool contextBreak=false;
      if(EfficientSegmentByContext)
      {
         if(gActiveEfficientPeriods[symIdx].lastLiquidityClass>=0 && item.liquidityClass != gActiveEfficientPeriods[symIdx].lastLiquidityClass)
            contextBreak=true;
         if(gActiveEfficientPeriods[symIdx].lastRegime>=0 && item.regime != gActiveEfficientPeriods[symIdx].lastRegime)
            contextBreak=true;
         if(gActiveEfficientPeriods[symIdx].lastStructureClass>=0 && item.structureContextClass != gActiveEfficientPeriods[symIdx].lastStructureClass)
            contextBreak=true;
      }

      if(contextBreak)
         gActiveEfficientPeriods[symIdx].contextBreakCount++;
      else
         gActiveEfficientPeriods[symIdx].contextBreakCount=0;

      double elapsedMinD = MathMax((double)elapsedMin, 1.0);
      double curEffPerMin = gActiveEfficientPeriods[symIdx].rewardTotal / elapsedMinD;
      bool degrade = (elapsedMin >= 30 && curEffPerMin < EfficientSegmentMinPerMin);
      bool ddBreak = (gActiveEfficientPeriods[symIdx].ddMax > EfficientPeriodMaxDDPct);

      if(elapsedMin >= EfficientPeriodWindowMinutes ||
         gActiveEfficientPeriods[symIdx].contextBreakCount >= EfficientContextBreakTolerance ||
         degrade || ddBreak)
      {
         FinalizeActiveEfficientPeriod(symIdx, 0, item.liquidityClass, item.regime, item.structureContextClass);
      }
   }

   EnsureActiveEfficientPeriod(symIdx, nowT);
   gActiveEfficientPeriods[symIdx].lastTime = nowT;
   gActiveEfficientPeriods[symIdx].rewardTotal += item.reward;
   gActiveEfficientPeriods[symIdx].ddMax = MathMax(gActiveEfficientPeriods[symIdx].ddMax, GetSymbolFloatingDDPct(symIdx));
   gActiveEfficientPeriods[symIdx].tradeCount++;
   if(item.addDepthClass==0 && item.reward>0.0) gActiveEfficientPeriods[symIdx].oneRoundCount++;
   gActiveEfficientPeriods[symIdx].addCountTotal += MathMax(item.addDepthClass,0);
   gActiveEfficientPeriods[symIdx].lastLiquidityClass = item.liquidityClass;
   gActiveEfficientPeriods[symIdx].lastRegime = item.regime;
   gActiveEfficientPeriods[symIdx].lastStructureClass = item.structureContextClass;
   int n=ArraySize(gActiveEfficientPeriods[symIdx].replayItemIndexes);
   ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,n+1);
   gActiveEfficientPeriods[symIdx].replayItemIndexes[n]=replayIdx;
}

int ClassifyReplayZoneContext(const string symbol,
                              const double px,
                              const double avgPx,
                              const double entryPx)
{
   if(!UseZoneCandleBranch) return (int)MEM_ZONE_NONE;

   double atr = MathMax(GetATRValueTF(symbol, TF_EXEC, 14, 1), 1e-8);

   SwingDerivedZone demands[], supplies[];
   int dc=0, sc=0;
   if(!BuildSwingDerivedZonesForTF(symbol, TF_EXEC, demands, dc, supplies, sc))
      return (int)MEM_ZONE_NONE;

   int bestD=-1, bestS=-1;
   bool hasD = SelectBestSwingZone(demands, dc, symbol, TF_EXEC, px, avgPx, entryPx, atr, bestD);
   bool hasS = SelectBestSwingZone(supplies, sc, symbol, TF_EXEC, px, avgPx, entryPx, atr, bestS);

   double nearBuf = 0.35;
   if(hasD && bestD>=0)
   {
      bool inside = (px >= demands[bestD].low && px <= demands[bestD].high);
      double dist = DistanceToZoneBoundaryAtr(demands[bestD], px, atr);
      if(inside) return (int)MEM_ZONE_INSIDE_DEMAND;
      if(dist <= nearBuf) return (int)MEM_ZONE_NEAR_DEMAND;
   }
   if(hasS && bestS>=0)
   {
      bool inside = (px >= supplies[bestS].low && px <= supplies[bestS].high);
      double dist = DistanceToZoneBoundaryAtr(supplies[bestS], px, atr);
      if(inside) return (int)MEM_ZONE_INSIDE_SUPPLY;
      if(dist <= nearBuf) return (int)MEM_ZONE_NEAR_SUPPLY;
   }
   return (int)MEM_ZONE_NONE;
}

int ClassifyReplayStructureContext(const string symbol)
{
   if(!UseStructureBranch) return (int)MEM_STRUCT_MIXED;
   double atr = MathMax(GetATRValueTF(symbol, TF_LONG, 14, 1), 1e-8);
   SwingPointMem swings[];
   int swingCount=0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   bool ok = BuildSwingMemoryForTF(symbol, TF_LONG, swings, swingCount) &&
             ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);
   if(!ok)
   {
      FallbackStructureRefs(symbol, TF_LONG, lastHigh, prevHigh, lastLow, prevLow);
   }
   double tendency = SwingTendencyFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr);
   double lastAmp = MathMax(lastHigh.price - lastLow.price, 0.0);
   double prevAmp = MathMax(prevHigh.price - prevLow.price, 0.0);
   double compExp = Clamp(SafeDiv(lastAmp, MathMax(prevAmp,1e-8), 1.0) - 1.0, -2.0, 2.0) / 2.0;

   if(MathAbs(tendency) >= 0.20)
      return (tendency > 0.0 ? (int)MEM_STRUCT_TREND_UP : (int)MEM_STRUCT_TREND_DOWN);
   if(compExp <= -0.20) return (int)MEM_STRUCT_COMPRESSION;
   if(compExp >= 0.20) return (int)MEM_STRUCT_EXPANSION;
   return (int)MEM_STRUCT_MIXED;
}

void FillReplayMetadata(ReplayItem &item,
                        const int symIdx,
                        const int regime,
                        const double &state[],
                        const int action,
                        const double reward,
                        const bool done)
{
   item.replayBankType = (int)REPLAY_BANK_RECENT;
   item.episodeId = GetCurrentEpisodeIdForSymbol(symIdx);
   item.patternId = 0;
   item.regimeId = (long)regime;
   item.periodId = 0;
   item.eventTime = TimeCurrent();

   string symbol = ((symIdx>=0 && symIdx<gSymbolCount) ? gSymbols[symIdx] : _Symbol);
   double px = GetCurrentMidPriceForSymbol(symbol);
   double avgPx = ((symIdx>=0 && symIdx<gSymbolCount && gBasketAvgValid[symIdx]) ? gBasketAvgPriceCache[symIdx] : px);
   double entryPx = GetLastEntryPriceFromCache(symIdx, px);

   item.basketStateClass    = ClassifyReplayBasketState(symIdx);
   item.addDepthClass       = ClassifyReplayAddDepth(symIdx);
   item.dangerClass         = ClassifyReplayDanger(symIdx, reward);
   item.zoneContextClass    = ClassifyReplayZoneContext(symbol, px, avgPx, entryPx);
   item.structureContextClass = ClassifyReplayStructureContext(symbol);
   item.liquidityClass      = ClassifyReplayLiquidity(symbol);
}

void ReplayBankStorePush(ReplayBankStore &bank, const ReplayItem &item, const int capacity)
{
   if(capacity<=0) return;
   bank.maxCount = capacity;
   int n = ArraySize(bank.items);
   ArrayResize(bank.items, n+1);
   bank.items[n] = item;
   if(ArraySize(bank.items) > capacity)
      ArrayRemove(bank.items, 0, 1);
   bank.count = ArraySize(bank.items);
}

void DeepBasketReplayStorePush(DeepBasketReplayStore &bank, const ReplayItem &item, const int capacity)
{
   if(capacity<=0) return;
   bank.maxItems = capacity;
   int n = ArraySize(bank.items);
   ArrayResize(bank.items, n+1);
   bank.items[n] = item;
   if(ArraySize(bank.items) > capacity)
      ArrayRemove(bank.items, 0, 1);
   bank.itemCount = ArraySize(bank.items);
}

void EfficientReplayStorePush(EfficientReplayStore &bank, const ReplayItem &item, const int capacity)
{
   if(capacity<=0) return;
   bank.maxItems = capacity;
   int n = ArraySize(bank.items);
   ArrayResize(bank.items, n+1);
   bank.items[n] = item;
   if(ArraySize(bank.items) > capacity)
      ArrayRemove(bank.items, 0, 1);
   bank.itemCount = ArraySize(bank.items);
}

double ReplayRiskyProfitScore(const ReplayItem &item)
{
   if(item.reward <= 0.0) return 0.0;

   double score = 0.0;
   if(item.addDepthClass >= 1) score += 0.55 + 0.35 * MathMin(item.addDepthClass, 3);
   if(item.basketStateClass >= 2) score += 0.45;
   if(item.dangerClass > 0) score += 0.55 * MathMin(item.dangerClass, 2);
   if(item.action != 0) score += 0.10;
   return score;
}

bool ReplayIsRiskyProfit(const ReplayItem &item)
{
   return (item.reward > 0.0 && ReplayRiskyProfitScore(item) >= ReplayRiskyProfitThreshold);
}

bool ReplayIsEfficientCandidate(const ReplayItem &item)
{
   if(item.reward <= 0.0) return false;
   if(item.dangerClass > 0) return false;
   if(item.addDepthClass > 0) return false;
   if(item.basketStateClass > 1) return false;
   return true;
}

bool ReplayIsDeepAntiPattern(const ReplayItem &item)
{
   bool basketHeavy = (item.addDepthClass >= 1 || item.basketStateClass >= 2);
   if(!basketHeavy) return false;
   if(item.reward < 0.0) return true;
   return ReplayIsRiskyProfit(item);
}

bool ReplayIsDangerAntiPattern(const ReplayItem &item)
{
   if(item.dangerClass >= 2 && item.reward <= 0.0) return true;
   if(item.dangerClass >= 1 && ReplayIsRiskyProfit(item)) return true;
   if(item.reward <= -1.5 && item.dangerClass >= 1) return true;
   return false;
}

double ReplayEfficiencyProgress()
{
   double eff = (double)ArraySize(gEfficientReplayBank.items) + 2.0 * (double)ArraySize(gEfficientReplayBank.periods);
   double anti = (double)ArraySize(gDangerReplayBank.items) + (double)ArraySize(gDeepBasketReplayBank.items) + 2.0 * (double)ArraySize(gDeepBasketReplayBank.sequences);
   return Clamp(SafeDiv(eff, eff + anti + 1.0, 0.0), 0.0, 1.0);
}

double ReplayAntiPatternPressure()
{
   double eff = (double)ArraySize(gEfficientReplayBank.items) + 2.0 * (double)ArraySize(gEfficientReplayBank.periods);
   double anti = (double)ArraySize(gDangerReplayBank.items) + (double)ArraySize(gDeepBasketReplayBank.items) + 2.0 * (double)ArraySize(gDeepBasketReplayBank.sequences);
   return Clamp(SafeDiv(anti, MathMax(1.0, eff), 0.0), 0.0, 3.0);
}

double ReplayRewardRecencyWeight(const datetime t)
{
   if(t<=0) return 1.0;
   if(ReplayRewardHalfLifeDays <= 0.0) return 1.0;
   double ageSec = (double)(TimeCurrent() - t);
   if(ageSec <= 0.0) return 1.0;
   double halfSec = ReplayRewardHalfLifeDays * 86400.0;
   if(halfSec <= 1.0) return 1.0;
   return Clamp(MathExp(-0.6931471805599453 * (ageSec / halfSec)), 0.10, 1.0);
}

double ReplayRewardContextWeight(const ReplayItem &it,
                                 const int symIdx,
                                 const int basketDir,
                                 const int positionsCount,
                                 const int regimeType,
                                 const int structureType,
                                 const int liquidityType)
{
   if(it.symIdx != symIdx) return 0.0;

   double w = 0.15;
   if(it.regime == regimeType) w += 0.22;
   else if(MathAbs(it.regime - regimeType) == 1) w += 0.10;
   if(it.structureContextClass == structureType) w += 0.16;
   if(it.liquidityClass == liquidityType) w += 0.12;

   int targetAction = (basketDir>0 ? 1 : (basketDir<0 ? 2 : -1));
   if(targetAction>0 && it.action == targetAction) w += 0.10;

   if(positionsCount <= 1 && it.addDepthClass <= 0) w += 0.07;
   if(positionsCount > 1 && it.addDepthClass >= 1) w += 0.10;

   w *= ReplayRewardRecencyWeight(it.eventTime);
   return w;
}

double ReplayRewardDangerScore(const ReplayItem &it)
{
   double s = Clamp(MathMax(0.0, -it.reward), 0.0, 3.0);
   s += 0.35 * (double)MathMax(0, it.dangerClass);
   s += 0.18 * (double)MathMax(0, it.addDepthClass);
   if(ReplayIsRiskyProfit(it)) s += 0.85;
   return Clamp(s, 0.0, 5.0);
}

double ReplayRewardDeepScore(const ReplayItem &it)
{
   double s = 0.25 * (double)MathMax(0, it.addDepthClass);
   s += 0.18 * (double)MathMax(0, it.basketStateClass);
   if(it.reward < 0.0) s += Clamp(-it.reward, 0.0, 3.0);
   if(ReplayIsRiskyProfit(it)) s += 0.90;
   return Clamp(s, 0.0, 5.0);
}

double ReplayRewardEfficientScore(const ReplayItem &it)
{
   double s = Clamp(it.reward, 0.0, 3.0);
   if(ReplayIsEfficientCandidate(it)) s += 0.75;
   if(it.addDepthClass <= 0) s += 0.20;
   if(it.dangerClass <= 0) s += 0.20;
   if(it.reward <= 0.0) s *= 0.25;
   return Clamp(s, 0.0, 5.0);
}

void ComputeReplayRewardProfile(const int symIdx,
                                const int basketDir,
                                const int positionsCount,
                                double &dangerRiskOut,
                                double &deepRiskOut,
                                double &efficientOut,
                                double &recentCautionOut)
{
   dangerRiskOut=0.0;
   deepRiskOut=0.0;
   efficientOut=0.0;
   recentCautionOut=0.0;

   if(!UseReplayAwareReward) return;
   if(symIdx<0 || symIdx>=gSymbolCount) return;

   string symbol = gSymbols[symIdx];
   int regimeType=0, structureType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, structureType, liquidityType, sessionType, atrRatio);
   int scan = MathMax(50, ReplayRewardScanLimit);

   double sumDanger=0.0, accDanger=0.0;
   int nd=ArraySize(gDangerReplayBank.items);
   int sd=MathMax(0, nd-scan);
   for(int i=sd;i<nd;i++)
   {
      double w = ReplayRewardContextWeight(gDangerReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      accDanger += w * ReplayRewardDangerScore(gDangerReplayBank.items[i]);
      sumDanger += w;
   }
   if(sumDanger>1e-9) dangerRiskOut = Clamp(accDanger / sumDanger, 0.0, 4.0);

   double sumDeep=0.0, accDeep=0.0;
   int nb=ArraySize(gDeepBasketReplayBank.items);
   int sb=MathMax(0, nb-scan);
   for(int i=sb;i<nb;i++)
   {
      double w = ReplayRewardContextWeight(gDeepBasketReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      if(positionsCount>1 && gDeepBasketReplayBank.items[i].addDepthClass>=1) w *= 1.15;
      accDeep += w * ReplayRewardDeepScore(gDeepBasketReplayBank.items[i]);
      sumDeep += w;
   }
   int ns=ArraySize(gDeepBasketReplayBank.sequences);
   int ss=MathMax(0, ns-scan);
   for(int i=ss;i<ns;i++)
   {
      if(gDeepBasketReplayBank.sequences[i].symIdx != symIdx) continue;
      double w = 0.20;
      if(gDeepBasketReplayBank.sequences[i].basketDir == basketDir) w += 0.22;
      if(positionsCount>1) w += 0.12;
      w *= ReplayRewardRecencyWeight(gDeepBasketReplayBank.sequences[i].endTime);

      double s = Clamp(gDeepBasketReplayBank.sequences[i].maxDD / MathMax(DangerReplayDDThresholdPct, 1e-6), 0.0, 4.0);
      s += 0.25 * (double)MathMax(0, gDeepBasketReplayBank.sequences[i].addCount);
      s += 0.18 * (double)MathMax(0, gDeepBasketReplayBank.sequences[i].maxPositions - 1);
      if(gDeepBasketReplayBank.sequences[i].finalReward < 0.0) s += Clamp(-gDeepBasketReplayBank.sequences[i].finalReward, 0.0, 2.0);
      if(gDeepBasketReplayBank.sequences[i].finalReward > 0.0 &&
         (gDeepBasketReplayBank.sequences[i].addCount >= 2 || gDeepBasketReplayBank.sequences[i].maxDD >= 0.5*DangerReplayDDThresholdPct))
         s += 0.75;

      accDeep += w * Clamp(s, 0.0, 5.0);
      sumDeep += w;
   }
   if(sumDeep>1e-9) deepRiskOut = Clamp(accDeep / sumDeep, 0.0, 5.0);

   double sumEff=0.0, accEff=0.0;
   int ne=ArraySize(gEfficientReplayBank.items);
   int se=MathMax(0, ne-scan);
   for(int i=se;i<ne;i++)
   {
      double w = ReplayRewardContextWeight(gEfficientReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      accEff += w * ReplayRewardEfficientScore(gEfficientReplayBank.items[i]);
      sumEff += w;
   }
   int np=ArraySize(gEfficientReplayBank.periods);
   int sp=MathMax(0, np-scan);
   for(int i=sp;i<np;i++)
   {
      if(gEfficientReplayBank.periods[i].symIdx != symIdx) continue;
      double w = 0.18;
      if(gEfficientReplayBank.periods[i].regimeType == regimeType) w += 0.18;
      else if(MathAbs(gEfficientReplayBank.periods[i].regimeType - regimeType) == 1) w += 0.08;
      if(gEfficientReplayBank.periods[i].patternType == structureType) w += 0.14;
      if(gEfficientReplayBank.periods[i].liquidityType == liquidityType) w += 0.10;
      if(positionsCount <= 1) w += 0.10;
      w *= ReplayRewardRecencyWeight(gEfficientReplayBank.periods[i].endTime);

      double s = Clamp(gEfficientReplayBank.periods[i].rewardEfficiency, -2.0, 2.0);
      s += 0.20 * Clamp((double)gEfficientReplayBank.periods[i].oneRoundCount, 0.0, 6.0);
      s -= 0.35 * Clamp(gEfficientReplayBank.periods[i].ddMax / MathMax(EfficientPeriodMaxDDPct, 1e-6), 0.0, 4.0);
      s -= 0.15 * Clamp((double)gEfficientReplayBank.periods[i].addCountTotal, 0.0, 8.0);
      accEff += w * Clamp(s, -2.0, 3.0);
      sumEff += w;
   }
   if(sumEff>1e-9) efficientOut = Clamp(accEff / sumEff, -1.0, 3.0);

   double sumRecent=0.0, accRecent=0.0;
   int nr=ArraySize(gRecentReplayBank.items);
   int sr=MathMax(0, nr-scan);
   for(int i=sr;i<nr;i++)
   {
      double w = ReplayRewardContextWeight(gRecentReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      double c = 0.0;
      if(ReplayIsDangerAntiPattern(gRecentReplayBank.items[i])) c += 0.70 * ReplayRewardDangerScore(gRecentReplayBank.items[i]);
      if(ReplayIsDeepAntiPattern(gRecentReplayBank.items[i]))   c += 0.85 * ReplayRewardDeepScore(gRecentReplayBank.items[i]);
      if(ReplayIsEfficientCandidate(gRecentReplayBank.items[i])) c -= 0.45 * ReplayRewardEfficientScore(gRecentReplayBank.items[i]);
      accRecent += w * c;
      sumRecent += w;
   }
   if(sumRecent>1e-9) recentCautionOut = Clamp(accRecent / sumRecent, -1.5, 3.0);
}

string BuildReplayDiagnosticsText(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return "";
   double effProg = ReplayEfficiencyProgress();
   double antiP = ReplayAntiPatternPressure();
   double risk = CurrentReplayRiskBias(symIdx);
   return StringFormat("Replay main=%d recent=%d danger=%d deep=%d eff=%d seq=%d per=%d | new R/D/DP/E=%d/%d/%d/%d | prog=%.2f anti=%.2f risk=%.2f",
                       ArraySize(gReplay),
                       ArraySize(gRecentReplayBank.items),
                       ArraySize(gDangerReplayBank.items),
                       ArraySize(gDeepBasketReplayBank.items),
                       ArraySize(gEfficientReplayBank.items),
                       ArraySize(gDeepBasketReplayBank.sequences),
                       ArraySize(gEfficientReplayBank.periods),
                       gReplayDiagWindowRecentAdds[symIdx],
                       gReplayDiagWindowDangerAdds[symIdx],
                       gReplayDiagWindowDeepAdds[symIdx],
                       gReplayDiagWindowEfficientAdds[symIdx],
                       effProg, antiP, risk);
}

void MaybePrintReplayDiagnostics(const int symIdx)
{
   if(!UseReplayDiagnostics || !ReplayDiagnosticsToLog) return;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   gReplayDiagBarsSincePrint[symIdx]++;
   int every = MathMax(1, ReplayDiagnosticsPrintEveryBars);
   if(gReplayDiagBarsSincePrint[symIdx] < every) return;

   Print(gSymbols[symIdx], " | ", BuildReplayDiagnosticsText(symIdx));
   gReplayDiagBarsSincePrint[symIdx]=0;
   gReplayDiagWindowRecentAdds[symIdx]=0;
   gReplayDiagWindowDangerAdds[symIdx]=0;
   gReplayDiagWindowDeepAdds[symIdx]=0;
   gReplayDiagWindowEfficientAdds[symIdx]=0;
}

bool EfficientPeriodQualifies(const EfficientPeriodRef &pr)
{
   if(pr.rewardEfficiency < EfficientPeriodMinRewardEfficiency) return false;
   if(pr.ddMax > EfficientPeriodMaxDDPct) return false;
   if(pr.addCountTotal > MathMax(1, pr.oneRoundCount + 1)) return false;
   if(pr.oneRoundCount <= 0 && pr.addCountTotal > 0) return false;
   return true;
}

bool DeepBasketSequenceQualifies(const BasketSequenceRef &seq)
{
   bool riskyProfit = (seq.finalReward > 0.0 &&
                       (seq.maxDD >= 0.5*DangerReplayDDThresholdPct || seq.addCount >= 2 || seq.maxPositions > 2));
   if(seq.finalReward < 0.0) return true;
   if(seq.maxPositions > 2) return true;
   if(seq.addCount >= 1 && seq.maxDD >= MathMax(0.0125, 0.50*DangerReplayDDThresholdPct)) return true;
   if(riskyProfit) return true;
   return false;
}

int ChooseReplayBankType(const ReplayItem &item)
{
   if(ReplayIsDeepAntiPattern(item))
      return (int)REPLAY_BANK_DEEP_BASKET;
   if(ReplayIsDangerAntiPattern(item))
      return (int)REPLAY_BANK_DANGER;
   if(ReplayIsEfficientCandidate(item))
      return (int)REPLAY_BANK_EFFICIENT;
   if(item.reward < 0.0 && item.dangerClass >= 1)
      return (int)REPLAY_BANK_DANGER;
   return (int)REPLAY_BANK_RECENT;
}

void RouteReplayItemToBanks(ReplayItem &item,const int replayIdx)
{
   ReplayBankStorePush(gRecentReplayBank, item, RecentReplayCapacity);
   if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS)
      gReplayDiagWindowRecentAdds[item.symIdx]++;

   int bankType = ChooseReplayBankType(item);
   item.replayBankType = bankType;

   if(bankType == (int)REPLAY_BANK_DANGER && UseDangerReplayBank)
   {
      ReplayBankStorePush(gDangerReplayBank, item, DangerReplayCapacity);
      if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS) gReplayDiagWindowDangerAdds[item.symIdx]++;
   }
   else if(bankType == (int)REPLAY_BANK_DEEP_BASKET && UseDeepBasketReplayBank)
   {
      DeepBasketReplayStorePush(gDeepBasketReplayBank, item, DeepBasketReplayCapacity);
      if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS) gReplayDiagWindowDeepAdds[item.symIdx]++;
   }
   else if(bankType == (int)REPLAY_BANK_EFFICIENT && UseEfficientReplayBank)
   {
      EfficientReplayStorePush(gEfficientReplayBank, item, EfficientReplayCapacity);
      if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS) gReplayDiagWindowEfficientAdds[item.symIdx]++;
   }

   DangerBrainObserveReplayBankItem(item);

   if(item.episodeId>0)
      AppendReplayIndexToActiveEpisode(item.symIdx, replayIdx, item.reward);

   UpdateActiveEfficientPeriodWithItem(item, replayIdx);
}

void ReplayPush(const int symIdx,
                const int regime,
                const double &state[],
                const int action,
                const double reward,
                const double &nextState[],
                const bool done)
{
   if(!UseReplayBuffer) return;
   if(action<0 || action>=ActionCount) return;

   ReplayItem item;
   item.symIdx=symIdx;
   item.regime=regime;
   item.action=action;
   item.reward=reward;
   item.done=done;
   item.priority=MathAbs(reward) + ReplayPriorityReward(item.reward);
   if(item.priority<0.01) item.priority=0.01;

   ArrayResize(item.state,ArraySize(state));
   for(int i=0;i<ArraySize(state);i++) item.state[i]=state[i];

   ArrayResize(item.nextState,ArraySize(nextState));
   for(int i=0;i<ArraySize(nextState);i++) item.nextState[i]=nextState[i];

   FillReplayMetadata(item, symIdx, regime, state, action, reward, done);

   int n=ArraySize(gReplay);
   ArrayResize(gReplay,n+1);
   gReplay[n]=item;

   RouteReplayItemToBanks(gReplay[n], n);
   gReplayPendingTrainCount++;

   if(ArraySize(gReplay)>ReplayCapacity)
      ArrayRemove(gReplay,0,1);
}

double ReplayPriorityReward(const double reward)
{
   return ReplayPriorityRewardK * MathAbs(reward);
}

bool ReplaySampleIndex(int &idxOut)
{
   idxOut=-1;
   int n=ArraySize(gReplay);
   if(n<=0) return false;

   double sumP=0.0;
   for(int i=0;i<n;i++) sumP += MathMax(0.0001, gReplay[i].priority);
   if(sumP<=1e-12) return false;

   double r=((double)MathRand()/32767.0)*sumP;
   double c=0.0;
   for(int i=0;i<n;i++)
   {
      c += MathMax(0.0001, gReplay[i].priority);
      if(r<=c){ idxOut=i; return true; }
   }

   idxOut=n-1;
   return true;
}

double DangerReplaySampleWeight(const ReplayItem &it)
{
   double w = MathMax(0.0001, it.priority);
   if(it.reward < 0.0) w *= (1.0 + 0.20 * MathMin(-it.reward, 3.0));
   if(it.addDepthClass <= 1) w *= DangerReplayEarlyStateBoost;
   if(it.addDepthClass >= 3) w *= DangerReplayLateRescueWeight;
   if(ReplayIsRiskyProfit(it)) w *= 1.10;
   w *= (1.0 + 1.15*Clamp(it.painSeverity,0.0,1.0) + 0.45*Clamp(it.recurrenceScore,0.0,1.0) + 0.30*Clamp(it.regimeBreakScore,0.0,1.0));
   return w;
}

double DeepBasketSampleWeight(const ReplayItem &it)
{
   double w = MathMax(0.0001, it.priority);
   if(it.addDepthClass<=1 && it.episodeId>0) w *= DeepSequenceEarlyBoost;
   else if(it.addDepthClass>=3) w *= DeepReplayLateRescueWeight;
   if(it.reward < 0.0) w *= (1.0 + 0.15 * MathMin(-it.reward, 3.0));
   if(ReplayIsRiskyProfit(it)) w *= 1.15;
   if(it.done && it.reward > 0.0) w *= 0.85;
   w *= (1.0 + 1.00*Clamp(it.painSeverity,0.0,1.0) + 0.55*Clamp(it.recurrenceScore,0.0,1.0));
   return w;
}

double EfficientReplaySampleWeight(const ReplayItem &it)
{
   double w = MathMax(0.0001, it.priority);
   if(ReplayIsEfficientCandidate(it)) w *= 1.35;
   else if(it.reward > 0.0 && it.addDepthClass<=1 && it.dangerClass==0) w *= 1.10;
   if(it.dangerClass>0) w *= 0.65;
   if(it.reward<=0.0) w *= 0.25;
   return w;
}

bool ReplaySampleFromArrayWeighted(const ReplayItem &arr[], ReplayItem &itemOut, int &idxOut, const int source)
{
   idxOut=-1;
   int n=ArraySize(arr);
   if(n<=0) return false;

   double sumP=0.0;
   for(int i=0;i<n;i++)
   {
      double w=MathMax(0.0001, arr[i].priority);
      if(source==2) w = DangerReplaySampleWeight(arr[i]);
      else if(source==3) w = DeepBasketSampleWeight(arr[i]);
      else if(source==4) w = EfficientReplaySampleWeight(arr[i]);
      sumP += MathMax(0.0001, w);
   }
   if(sumP<=1e-12) return false;

   double r=((double)MathRand()/32767.0)*sumP;
   double c=0.0;
   for(int i=0;i<n;i++)
   {
      double w=MathMax(0.0001, arr[i].priority);
      if(source==2) w = DangerReplaySampleWeight(arr[i]);
      else if(source==3) w = DeepBasketSampleWeight(arr[i]);
      else if(source==4) w = EfficientReplaySampleWeight(arr[i]);
      c += MathMax(0.0001, w);
      if(r<=c){ idxOut=i; itemOut=arr[i]; return true; }
   }

   idxOut=n-1;
   itemOut=arr[n-1];
   return true;
}

enum ReplaySampleSource
{
   REPLAY_SRC_MAIN=0,
   REPLAY_SRC_RECENT=1,
   REPLAY_SRC_DANGER=2,
   REPLAY_SRC_DEEP=3,
   REPLAY_SRC_EFFICIENT=4
};

bool SampleReplayItemMixed(ReplayItem &itemOut,int &srcOut,int &idxOut)
{
   srcOut=REPLAY_SRC_MAIN;
   idxOut=-1;

   if(!UseBankAwareReplaySampling)
   {
      if(!ReplaySampleIndex(idxOut)) return false;
      itemOut=gReplay[idxOut];
      return true;
   }

   double wMain = MathMax(0.0, ReplaySampleWeightMain);
   double wRecent = (UseRecentReplayBank?MathMax(0.0, ReplaySampleWeightRecent):0.0);
   double wDanger = (UseDangerReplayBank?MathMax(0.0, ReplaySampleWeightDanger):0.0);
   double wDeep = (UseDeepBasketReplayBank?MathMax(0.0, ReplaySampleWeightDeepBasket):0.0);
   double wEff = (UseEfficientReplayBank?MathMax(0.0, ReplaySampleWeightEfficient):0.0);

   double replayRisk = 0.0;
   replayRisk = MathMax(replayRisk, SafeDiv(CurrentReplayDDPct(), MathMax(DangerReplayDDThresholdPct, 1e-6), 0.0));
   double dangerEMA = 0.0, deepEMA = 0.0;
   for(int s=0;s<gSymbolCount;s++)
   {
      dangerEMA = MathMax(dangerEMA, gDangerReplaySeverityEMA[s]);
      deepEMA   = MathMax(deepEMA,   gDeepBasketReplaySeverityEMA[s]);
   }
   replayRisk = MathMax(replayRisk, Clamp(dangerEMA / 3.0, 0.0, 2.0));
   replayRisk = MathMax(replayRisk, Clamp(deepEMA   / 3.0, 0.0, 2.0));

   if(replayRisk > 1.0)
   {
      double boost = Clamp(replayRisk - 1.0, 0.0, 1.0);
      wDanger *= (1.0 + 0.75 * boost);
      wDeep   *= (1.0 + 0.65 * boost);
      wEff    *= (1.0 - 0.30 * boost);
      wRecent *= (1.0 + 0.15 * boost);
      wMain   *= (1.0 - 0.10 * boost);
   }
   else
   {
      double calm = Clamp(1.0 - replayRisk, 0.0, 1.0);
      wEff    *= (1.0 + 0.50 * calm);
      wRecent *= (1.0 + 0.25 * calm);
   }

   double effProgress = ReplayEfficiencyProgress();
   double antiPressure = ReplayAntiPatternPressure();
   wEff    *= (1.0 + ReplayMatureEfficiencyBoost * effProgress);
   wRecent *= (1.0 + 0.20 * effProgress);
   wDanger *= MathMax(0.25, 1.0 - ReplayAntiPatternDecayScale * effProgress);
   wDeep   *= MathMax(0.25, 1.0 - ReplayAntiPatternDecayScale * effProgress);
   if(antiPressure > 1.0)
   {
      double p = Clamp(antiPressure - 1.0, 0.0, 1.5);
      wDanger *= (1.0 + 0.20 * p);
      wDeep   *= (1.0 + 0.25 * p);
   }

   bool okMain   = (ArraySize(gReplay)>0);
   bool okRecent = (ArraySize(gRecentReplayBank.items)>0);
   bool okDanger = (ArraySize(gDangerReplayBank.items)>0);
   bool okDeep   = (ArraySize(gDeepBasketReplayBank.items)>0);
   bool okEff    = (ArraySize(gEfficientReplayBank.items)>0);

   if(!okMain)   wMain=0.0;
   if(!okRecent) wRecent=0.0;
   if(!okDanger) wDanger=0.0;
   if(!okDeep)   wDeep=0.0;
   if(!okEff)    wEff=0.0;

   double wSum = wMain+wRecent+wDanger+wDeep+wEff;
   if(wSum<=1e-12)
   {
      if(!ReplaySampleIndex(idxOut)) return false;
      itemOut=gReplay[idxOut];
      srcOut=REPLAY_SRC_MAIN;
      return true;
   }

   double r=((double)MathRand()/32767.0)*wSum;
   double c=0.0;

   c += wMain;
   if(r<=c && okMain)
   {
      if(!ReplaySampleIndex(idxOut)) return false;
      itemOut=gReplay[idxOut];
      srcOut=REPLAY_SRC_MAIN;
      return true;
   }
   c += wRecent;
   if(r<=c && okRecent)
   {
      srcOut=REPLAY_SRC_RECENT;
      return ReplaySampleFromArrayWeighted(gRecentReplayBank.items, itemOut, idxOut, REPLAY_SRC_RECENT);
   }
   c += wDanger;
   if(r<=c && okDanger)
   {
      srcOut=REPLAY_SRC_DANGER;
      return ReplaySampleFromArrayWeighted(gDangerReplayBank.items, itemOut, idxOut, REPLAY_SRC_DANGER);
   }
   c += wDeep;
   if(r<=c && okDeep)
   {
      srcOut=REPLAY_SRC_DEEP;
      return ReplaySampleFromArrayWeighted(gDeepBasketReplayBank.items, itemOut, idxOut, REPLAY_SRC_DEEP);
   }
   if(okEff)
   {
      srcOut=REPLAY_SRC_EFFICIENT;
      return ReplaySampleFromArrayWeighted(gEfficientReplayBank.items, itemOut, idxOut, REPLAY_SRC_EFFICIENT);
   }

   if(!ReplaySampleIndex(idxOut)) return false;
   itemOut=gReplay[idxOut];
   srcOut=REPLAY_SRC_MAIN;
   return true;
}

void DangerBrainObserveReplayBankItem(const ReplayItem &item)
{
   if(!UseDangerBrainReplayHook) return;
   int s=item.symIdx;
   if(s<0 || s>=MAX_SYMBOLS) return;

   double sev = MathMax(MathAbs(item.reward), 0.10);
   sev *= (1.0 + 0.25*MathMax(0,item.dangerClass) + 0.10*MathMax(0,item.addDepthClass));
   sev *= (1.0 + 0.85*Clamp(item.painSeverity,0.0,1.0) + 0.35*Clamp(item.recurrenceScore,0.0,1.0));

   if(item.replayBankType == (int)REPLAY_BANK_DANGER)
   {
      gDangerReplaySeverityEMA[s] = (1.0-DangerReplayObsAlpha)*gDangerReplaySeverityEMA[s] + DangerReplayObsAlpha*sev;
      gDangerReplayHits[s]++;
      if(item.done && item.reward < 0.0 && gBadLabel[s] >= 0)
         LearnBadEpisodeSmart(s,false);
   }
   else if(item.replayBankType == (int)REPLAY_BANK_DEEP_BASKET)
   {
      gDeepBasketReplaySeverityEMA[s] = (1.0-DangerReplayObsAlpha)*gDeepBasketReplaySeverityEMA[s] + DangerReplayObsAlpha*sev;
      gDeepBasketReplayHits[s]++;
      if(item.done && item.reward < 0.0 && gBadLabel[s] >= 0)
         LearnBadEpisodeSmart(s,false);
   }
   else if(item.replayBankType == (int)REPLAY_BANK_EFFICIENT)
   {
      gDangerReplaySeverityEMA[s]     *= (1.0 - 0.50*DangerReplayObsAlpha);
      gDeepBasketReplaySeverityEMA[s] *= (1.0 - 0.55*DangerReplayObsAlpha);
      if(gDangerReplayHits[s] > 0) gDangerReplayHits[s]--;
      if(gDeepBasketReplayHits[s] > 0) gDeepBasketReplayHits[s]--;
   }
}


double QMemPositiveBoostScale = 1.00;
double QMemNegativePenaltyScale = 1.00;
bool   QMemUseNegativeUpdates = true;
double ReplayPriorityRewardK = 0.25;

//  Next part starts at: double CombinedSim(...)
double CombinedSim(const int symIdx,
                   const double &fp_now[], const double &mini_now[],
                   const double &dfp_now[], const bool haveDFP,
                   const double &dmini_now[], const bool haveDMini,
                   const ProtoEntry &p)
{
   double s_fp = CosSim(fp_now, p.features);
   double s = s_fp;

   if(UseMiniStateFingerprint && gMiniValid[symIdx] && ArraySize(p.stateMini)==MINI_DIM)
   {
      double s_m = CosSim(mini_now, p.stateMini);
      double w1 = MathMax(0.0, SimW_PriceFingerprint);
      double w2 = MathMax(0.0, SimW_MiniState);
      double den = w1+w2;
      if(den>1e-9) s = (w1*s_fp + w2*s_m)/den;
   }

   if(UseDeltaSimilarity && haveDFP && ArraySize(p.deltaFP)==6)
   {
      double s_dfp = CosSim(dfp_now, p.deltaFP);
      double w = MathMax(0.0, SimW_DeltaFingerprint);
      s += w * s_dfp;
   }
   if(UseDeltaSimilarity && haveDMini && ArraySize(p.deltaMini)==MINI_DIM)
   {
      double s_dm = CosSim(dmini_now, p.deltaMini);
      double w = MathMax(0.0, SimW_DeltaMiniState);
      s += w * s_dm;
   }

   double ageF = ProtoAgeFactor(p);
   s *= ageF;

   double u = (double)p.usedCount;
   if(u>0.0) s += 0.02 * MathLog(1.0 + u);

   return Clamp(s, -2.0, 2.0);
}

double ComputeExposureScore(const int symIdx)
{
   double eq=GetEAEquity();
   double dd=0.0;
   if(maxEquity>1e-9 && eq<maxEquity) dd=Clamp((maxEquity-eq)/maxEquity,0.0,1.0);

   double s=0.0;
   s += ExposureW_Positions * gPositionsCount[symIdx];
   s += ExposureW_DD * dd;
   return s;
}

int DangerPredict(const double &f_now[], const int symIdx, double &P_danger)
{
   double mini_now[];
   ArrayResize(mini_now, MINI_DIM);
   for(int k=0;k<MINI_DIM;k++) mini_now[k]=gMiniCache[symIdx][k];

   double dfp_now[];
   bool haveDFP=false;
   if(gFPPrevValid[symIdx])
   {
      double tmpCur[]; ArrayResize(tmpCur,6);
      double tmpPrev[]; ArrayResize(tmpPrev,6);
      for(int k=0;k<6;k++){ tmpCur[k]=f_now[k]; tmpPrev[k]=gFPPrev[symIdx][k]; }
      haveDFP = BuildDeltaVec(tmpCur, tmpPrev, 6, dfp_now);
   }

   double dmini_now[];
   bool haveDMini=false;
   if(UseMiniStateFingerprint && gMiniPrevValid[symIdx] && gMiniValid[symIdx])
   {
      double cur[]; ArrayResize(cur,MINI_DIM);
      double prev[]; ArrayResize(prev,MINI_DIM);
      for(int k=0;k<MINI_DIM;k++){ cur[k]=gMiniCache[symIdx][k]; prev[k]=gMiniPrev[symIdx][k]; }
      haveDMini = BuildDeltaVec(cur, prev, MINI_DIM, dmini_now);
   }

   double bestDanger=-1e9, bestSafe=-1e9;
   int bestDangerIdx=-1;

   int n=ArraySize(gProtos);
   for(int i=0;i<n;i++)
   {
      if(ArraySize(gProtos[i].features)!=ArraySize(f_now)) continue;
      double sim = CombinedSim(symIdx, f_now, mini_now, dfp_now, haveDFP, dmini_now, haveDMini, gProtos[i])
                  + RegimeContextSimilarityBonus(GetAtrRatioCached_Base(gSymbols[symIdx]), gProtos[i].atrRatio);
      if(gProtos[i].isDanger)
      {
         if(sim>bestDanger){ bestDanger=sim; bestDangerIdx=i; }
      }
      else bestSafe=MathMax(bestSafe, sim);
   }

   double simContrast=(bestDanger-bestSafe);
   double exposure=ComputeExposureScore(symIdx);
   double raw = ScoreW_SimContrast*simContrast + ScoreW_Exposure*exposure;

   P_danger=Sigmoid(raw);
   return bestDangerIdx;
}

void UpdateMode(const int symIdx, const double P_danger, const datetime now)
{
   BrainMode m=gMode[symIdx];
   datetime last=gModeLastChange[symIdx];
   bool cooldownOk=(last==0) || ((now-last) >= ModeCooldownMinutes*60);

   if(m==MODE_NORMAL)
   {
      if(P_danger>=gT2Enter){ gMode[symIdx]=MODE_DANGER; gModeLastChange[symIdx]=now; }
      else if(P_danger>=gT1Enter){ gMode[symIdx]=MODE_CAUTION; gModeLastChange[symIdx]=now; }
   }
   else if(m==MODE_CAUTION)
   {
      if(P_danger>=gT2Enter){ gMode[symIdx]=MODE_DANGER; gModeLastChange[symIdx]=now; }
      else if(P_danger<=gT1Exit){ gMode[symIdx]=MODE_NORMAL; gModeLastChange[symIdx]=now; }
   }
   else
   {
      if(cooldownOk && P_danger<=gT2Exit)
      {
         gMode[symIdx]=MODE_CAUTION;
         gModeLastChange[symIdx]=now;
      }
   }

   if(gMode[symIdx]==MODE_DANGER) gLRScale[symIdx]=LRScale_Danger;
   else if(gMode[symIdx]==MODE_CAUTION) gLRScale[symIdx]=LRScale_Caution;
   else gLRScale[symIdx]=1.0;
}

bool GetBestAdapterBiasSmart(const int symIdx, const double &f_now[], double &outBias[], double &simBest)
{
   simBest=-1e9;
   int best=-1;

   double mini_now[];
   ArrayResize(mini_now, MINI_DIM);
   for(int k=0;k<MINI_DIM;k++) mini_now[k]=gMiniCache[symIdx][k];

   double dfp_now[];
   bool haveDFP=false;
   if(gFPPrevValid[symIdx])
   {
      double tmpCur[]; ArrayResize(tmpCur,6);
      double tmpPrev[]; ArrayResize(tmpPrev,6);
      for(int k=0;k<6;k++){ tmpCur[k]=f_now[k]; tmpPrev[k]=gFPPrev[symIdx][k]; }
      haveDFP = BuildDeltaVec(tmpCur, tmpPrev, 6, dfp_now);
   }

   double dmini_now[];
   bool haveDMini=false;
   if(UseMiniStateFingerprint && gMiniPrevValid[symIdx] && gMiniValid[symIdx])
   {
      double cur[]; ArrayResize(cur,MINI_DIM);
      double prev[]; ArrayResize(prev,MINI_DIM);
      for(int k=0;k<MINI_DIM;k++){ cur[k]=gMiniCache[symIdx][k]; prev[k]=gMiniPrev[symIdx][k]; }
      haveDMini = BuildDeltaVec(cur, prev, MINI_DIM, dmini_now);
   }

   int n=ArraySize(gProtos);
   for(int i=0;i<n;i++)
   {
      if(!gProtos[i].isDanger) continue;
      if(ArraySize(gProtos[i].features)!=ArraySize(f_now)) continue;

      double sim = CombinedSim(symIdx, f_now, mini_now, dfp_now, haveDFP, dmini_now, haveDMini, gProtos[i])
                  + RegimeContextSimilarityBonus(GetAtrRatioCached_Base(gSymbols[symIdx]), gProtos[i].atrRatio);

      double bonus=0.0;
      if(gPositionsCount[symIdx]>0)
      {
         int curLabel  = gBadLabel[symIdx];
         int curBasket = gBadBasketDir[symIdx];
         int curTrend  = gBadTrendDir[symIdx];

         if(curLabel>=0 && gProtos[i].label==curLabel) bonus += 0.05;
         if(curBasket!=0 && gProtos[i].basketDir==curBasket) bonus += 0.03;
         if(curTrend!=0  && gProtos[i].trendDir==curTrend)   bonus += 0.03;
      }

      double score = sim + bonus;
      if(score>simBest){ simBest=score; best=i; }
   }
   if(best<0) return false;

   int aN=ArraySize(gProtos[best].adapterBias);
   if(aN<=0) return false;

   ArrayResize(outBias,aN);
   for(int a=0;a<aN;a++) outBias[a]=gProtos[best].adapterBias[a];

   gProtos[best].usedCount++;
   gProtos[best].lastUsed=TimeCurrent();
   return true;
}

double ComputeAlphaMix(const int symIdx, const double simBest)
{
   double p=gPDanger[symIdx];
   double sim=MathMax(0.0, MathMin(1.0, (simBest+1.0)*0.5));
   double a=(p - gT1Enter) / MathMax(1e-6, (1.0 - gT1Enter));
   a=Clamp(a,0.0,1.0);
   return a*sim;
}

bool SaveDangerMemoryFile(const string filename)
{
   int h=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   FileWriteInteger(h, 5);
   int n=ArraySize(gProtos);
   FileWriteInteger(h, n);

   for(int i=0;i<n;i++)
   {
      FileWriteInteger(h, gProtos[i].label);
      FileWriteInteger(h, gProtos[i].basketDir);
      FileWriteInteger(h, gProtos[i].trendDir);
      FileWriteDouble (h, gProtos[i].atrRatio);
      FileWriteLong   (h, (long)gProtos[i].created);

      FileWriteInteger(h, gProtos[i].isDanger ? 1 : 0);

      WriteDoubleArray(h, gProtos[i].features);
      WriteDoubleArray(h, gProtos[i].stateMini);
      WriteDoubleArray(h, gProtos[i].deltaFP);
      WriteDoubleArray(h, gProtos[i].deltaMini);
      WriteDoubleArray(h, gProtos[i].adapterBias);
      WriteDoubleArray(h, gProtos[i].qSnap);
      FileWriteDouble(h, gProtos[i].painMean);
      FileWriteDouble(h, gProtos[i].painCount);
      WriteDoubleArray(h, gProtos[i].macroSig);
      WriteDoubleArray(h, gProtos[i].microSig);
      FileWriteDouble(h, gProtos[i].deepBasketRate);
      FileWriteDouble(h, gProtos[i].regimeBreakRate);
      FileWriteDouble(h, gProtos[i].counterTrendFailureRate);
      FileWriteDouble(h, gProtos[i].reversalTrapRate);
      FileWriteDouble(h, gProtos[i].recoveryFailureRate);

      FileWriteDouble(h, gProtos[i].survivalScore);
      FileWriteInteger(h, (int)gProtos[i].usedCount);
      FileWriteLong(h, (long)gProtos[i].lastUsed);
   }

   FileClose(h);
   return true;
}

bool LoadDangerMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename, FILE_READ|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   int ver=FileReadInteger(h);
   if(ver!=1 && ver!=2 && ver!=3 && ver!=4 && ver!=5){ FileClose(h); return false; }

   int n=FileReadInteger(h);
   if(n<0 || n>50000){ FileClose(h); return false; }

   ArrayResize(gProtos,0);
   ArrayResize(gProtos,n);

   for(int i=0;i<n;i++)
   {
      if(ver>=2)
      {
         gProtos[i].label     = FileReadInteger(h);
         gProtos[i].basketDir = FileReadInteger(h);
         gProtos[i].trendDir  = FileReadInteger(h);
         gProtos[i].atrRatio  = FileReadDouble(h);
         gProtos[i].created   = (datetime)FileReadLong(h);
      }
      else
      {
         int oldType=FileReadInteger(h);
         gProtos[i].label=oldType;
         gProtos[i].basketDir=0;
         gProtos[i].trendDir=0;
         gProtos[i].atrRatio=1.0;
         gProtos[i].created=0;
      }

      gProtos[i].isDanger = (FileReadInteger(h)==1);

      ReadDoubleArray(h, gProtos[i].features);

      if(ver>=3)
      {
         ReadDoubleArray(h, gProtos[i].stateMini);
      }
      else ArrayResize(gProtos[i].stateMini, 0);

      if(ver>=4)
      {
         ReadDoubleArray(h, gProtos[i].deltaFP);
         ReadDoubleArray(h, gProtos[i].deltaMini);
      }
      else
      {
         ArrayResize(gProtos[i].deltaFP, 0);
         ArrayResize(gProtos[i].deltaMini, 0);
      }

      ReadDoubleArray(h, gProtos[i].adapterBias);

      if(ver>=2)
      {
         ReadDoubleArray(h, gProtos[i].qSnap);
      }
      else ArrayResize(gProtos[i].qSnap,0);

      if(ver>=5)
      {
         gProtos[i].painMean = FileReadDouble(h);
         gProtos[i].painCount = FileReadDouble(h);
         ReadDoubleArray(h, gProtos[i].macroSig);
         ReadDoubleArray(h, gProtos[i].microSig);
         gProtos[i].deepBasketRate = FileReadDouble(h);
         gProtos[i].regimeBreakRate = FileReadDouble(h);
         gProtos[i].counterTrendFailureRate = FileReadDouble(h);
         gProtos[i].reversalTrapRate = FileReadDouble(h);
         gProtos[i].recoveryFailureRate = FileReadDouble(h);
      }
      else
      {
         gProtos[i].painMean = 0.0;
         gProtos[i].painCount = 0.0;
         ArrayResize(gProtos[i].macroSig,0);
         ArrayResize(gProtos[i].microSig,0);
         gProtos[i].deepBasketRate = 0.0;
         gProtos[i].regimeBreakRate = 0.0;
         gProtos[i].counterTrendFailureRate = 0.0;
         gProtos[i].reversalTrapRate = 0.0;
         gProtos[i].recoveryFailureRate = 0.0;
      }

      gProtos[i].survivalScore = FileReadDouble(h);
      gProtos[i].usedCount = (uint)FileReadInteger(h);
      gProtos[i].lastUsed = (datetime)FileReadLong(h);
   }

   FileClose(h);
   return true;
}

void SeedTestProtos()
{
   if(ArraySize(gProtos)>0) return;

   ProtoEntry p;
   p.label=LBL_AGAINST_UPTREND;
   p.basketDir=-1;
   p.trendDir=+1;
   p.atrRatio=1.0;
   p.created=TimeCurrent();
   p.isDanger=true;

   ArrayResize(p.features,6);
   p.features[0]=0.6; p.features[1]=0.6; p.features[2]=0.1; p.features[3]=0.2; p.features[4]=0.1; p.features[5]=0.4;
   NormalizeVec(p.features);

   ArrayResize(p.stateMini,0);
   ArrayResize(p.deltaFP,0);
   ArrayResize(p.deltaMini,0);

   ArrayResize(p.adapterBias, ActionCount);
   for(int i=0;i<ActionCount;i++) p.adapterBias[i]=0.0;
   if(ActionCount>=1) p.adapterBias[0]= +0.4;
   if(ActionCount>=2) p.adapterBias[1]= -0.2;
   if(ActionCount>=3) p.adapterBias[2]= -0.6;

   ArrayResize(p.qSnap,0);
   p.survivalScore=1.0; p.usedCount=0; p.lastUsed=0;

   int n=ArraySize(gProtos);
   ArrayResize(gProtos,n+1); gProtos[n]=p;

   p.isDanger=false;
   p.label=LBL_AGAINST_DOWNTREND;
   p.basketDir=+1;
   p.trendDir=-1;
   if(ActionCount>=1) p.adapterBias[0]=0.0;
   if(ActionCount>=2) p.adapterBias[1]=0.0;
   if(ActionCount>=3) p.adapterBias[2]=0.0;
   n=ArraySize(gProtos);
   ArrayResize(gProtos,n+1); gProtos[n]=p;
}

bool SaveQMemoryFile(const string filename)
{
   int h=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   FileWriteInteger(h, 2);
   int n=ArraySize(gQMem);
   FileWriteInteger(h, n);

   for(int i=0;i<n;i++)
   {
      FileWriteInteger(h, gQMem[i].regime);
      FileWriteLong   (h, (long)gQMem[i].created);
      FileWriteLong   (h, (long)gQMem[i].lastUsed);
      FileWriteInteger(h, (int)gQMem[i].usedCount);

      WriteDoubleArray(h, gQMem[i].stateKey);
      WriteDoubleArray(h, gQMem[i].qVals);

      FileWriteDouble(h, gQMem[i].conf);
      FileWriteDouble(h, gQMem[i].score);
   }

   FileClose(h);
   return true;
}

bool LoadQMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;

   int h=FileOpen(filename, FILE_READ|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   int ver=FileReadInteger(h);
   if(ver!=1){ FileClose(h); return false; }

   int n=FileReadInteger(h);
   if(n<0 || n>200000){ FileClose(h); return false; }

   ArrayResize(gQMem,0);
   ArrayResize(gQMem,n);

   for(int i=0;i<n;i++)
   {
      gQMem[i].regime   = FileReadInteger(h);
      gQMem[i].created  = (datetime)FileReadLong(h);
      gQMem[i].lastUsed = (datetime)FileReadLong(h);
      gQMem[i].usedCount= (uint)FileReadInteger(h);

      ReadDoubleArray(h, gQMem[i].stateKey);
      ReadDoubleArray(h, gQMem[i].qVals);

      gQMem[i].conf  = FileReadDouble(h);
      gQMem[i].score = FileReadDouble(h);
   }

   FileClose(h);
   PruneQMemoryIfNeeded();
   return true;
}

bool SaveDDEventMemoryFile(const string filename)
{
   int h=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   FileWriteInteger(h, 1);

   int n=ArraySize(gDDEvents);
   FileWriteInteger(h, n);

   for(int i=0;i<n;i++)
   {
      FileWriteLong(h, (long)gDDEvents[i].created);
      FileWriteLong(h, (long)gDDEvents[i].triggerTime);
      FileWriteString(h, gDDEvents[i].symbol);
      FileWriteInteger(h, gDDEvents[i].magic);
      FileWriteInteger(h, gDDEvents[i].regimeAtTrigger);
      FileWriteInteger(h, gDDEvents[i].basketDirAtTrigger);
      FileWriteDouble(h, gDDEvents[i].ddAtTrigger);
      FileWriteInteger(h, gDDEvents[i].hardTrigger ? 1 : 0);
      FileWriteInteger(h, gDDEvents[i].completed ? 1 : 0);

      WriteDoubleArray(h, gDDEvents[i].triggerStateKey);
      WriteDoubleArray(h, gDDEvents[i].triggerQVals);
      FileWriteDouble(h, gDDEvents[i].peakDD);
      FileWriteInteger(h, gDDEvents[i].basketDepthMax);
      FileWriteDouble(h, gDDEvents[i].timeUnderWaterNorm);
      FileWriteDouble(h, gDDEvents[i].recoveryFailureScore);
      FileWriteDouble(h, gDDEvents[i].painSeverity);
      FileWriteInteger(h, gDDEvents[i].eventType);
      WriteDoubleArray(h, gDDEvents[i].macroSig);
      WriteDoubleArray(h, gDDEvents[i].microSig);

      int preN=ArraySize(gDDEvents[i].preBaskets);
      FileWriteInteger(h, preN);
      for(int b=0;b<preN;b++)
         WriteBasketSnapshot(h, gDDEvents[i].preBaskets[b]);

      int postN=ArraySize(gDDEvents[i].postBaskets);
      FileWriteInteger(h, postN);
      for(int b=0;b<postN;b++)
         WriteBasketSnapshot(h, gDDEvents[i].postBaskets[b]);

      int tickN=ArraySize(gDDEvents[i].ticks);
      FileWriteInteger(h, tickN);
      for(int t=0;t<tickN;t++)
         WriteTickTraceItem(h, gDDEvents[i].ticks[t]);
   }

   FileClose(h);
   return true;
}

bool LoadDDEventMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;

   int h=FileOpen(filename, FILE_READ|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   int ver=FileReadInteger(h);
   if(ver!=1){ FileClose(h); return false; }

   int n=FileReadInteger(h);
   if(n<0 || n>10000){ FileClose(h); return false; }

   ArrayResize(gDDEvents, n);

   for(int i=0;i<n;i++)
   {
      gDDEvents[i].created = (datetime)FileReadLong(h);
      gDDEvents[i].triggerTime = (datetime)FileReadLong(h);
      gDDEvents[i].symbol = FileReadString(h);
      gDDEvents[i].magic = FileReadInteger(h);
      gDDEvents[i].regimeAtTrigger = FileReadInteger(h);
      gDDEvents[i].basketDirAtTrigger = FileReadInteger(h);
      gDDEvents[i].ddAtTrigger = FileReadDouble(h);
      gDDEvents[i].hardTrigger = (FileReadInteger(h)==1);
      gDDEvents[i].completed = (FileReadInteger(h)==1);

      ReadDoubleArray(h, gDDEvents[i].triggerStateKey);
      ReadDoubleArray(h, gDDEvents[i].triggerQVals);

      int preN=FileReadInteger(h);
      ArrayResize(gDDEvents[i].preBaskets, preN);
      for(int b=0;b<preN;b++)
         ReadBasketSnapshot(h, gDDEvents[i].preBaskets[b]);

      int postN=FileReadInteger(h);
      ArrayResize(gDDEvents[i].postBaskets, postN);
      for(int b=0;b<postN;b++)
         ReadBasketSnapshot(h, gDDEvents[i].postBaskets[b]);

      int tickN=FileReadInteger(h);
      ArrayResize(gDDEvents[i].ticks, tickN);
      for(int t=0;t<tickN;t++)
         ReadTickTraceItem(h, gDDEvents[i].ticks[t]);
   }

   FileClose(h);
   return true;
}

void CloneDQN(const DQNNetwork &src, DQNNetwork &dst)
{
   dst.input_dim  = src.input_dim;
   dst.hidden_dim = src.hidden_dim;
   dst.hidden_dim2= src.hidden_dim2;
   dst.output_dim = src.output_dim;
   dst.fusion_dim = src.fusion_dim;

   dst.basket_h1    = src.basket_h1;
   dst.basket_h2    = src.basket_h2;
   dst.indicator_h1 = src.indicator_h1;
   dst.indicator_h2 = src.indicator_h2;
   dst.volatility_h1= src.volatility_h1;
   dst.volatility_h2= src.volatility_h2;
   dst.structure_h1 = src.structure_h1;
   dst.structure_h2 = src.structure_h2;
   dst.zone_h1      = src.zone_h1;
   dst.zone_h2      = src.zone_h2;

   #define CLONE_ARR(name) ArrayResize(dst.name, ArraySize(src.name)); for(int i=0;i<ArraySize(src.name);i++) dst.name[i]=src.name[i];
   CLONE_ARR(basket_W1); CLONE_ARR(basket_b1); CLONE_ARR(basket_W2); CLONE_ARR(basket_b2);
   CLONE_ARR(indicator_W1); CLONE_ARR(indicator_b1); CLONE_ARR(indicator_W2); CLONE_ARR(indicator_b2);
   CLONE_ARR(volatility_W1); CLONE_ARR(volatility_b1); CLONE_ARR(volatility_W2); CLONE_ARR(volatility_b2);
   CLONE_ARR(structure_W1); CLONE_ARR(structure_b1); CLONE_ARR(structure_W2); CLONE_ARR(structure_b2);
   CLONE_ARR(zone_W1); CLONE_ARR(zone_b1); CLONE_ARR(zone_W2); CLONE_ARR(zone_b2);
   CLONE_ARR(W1); CLONE_ARR(b1); CLONE_ARR(W2); CLONE_ARR(b2); CLONE_ARR(WV); CLONE_ARR(bV); CLONE_ARR(WA); CLONE_ARR(bA);
   CLONE_ARR(feat_mean); CLONE_ARR(feat_std);
   #undef CLONE_ARR
}

void SyncTargetNetFor(const int symIdx,const int regime)
{
   CloneDQN(gDQN[symIdx][regime], gTargetDQN[symIdx][regime]);
}

void SoftUpdateTargetNetFor(const int symIdx,const int regime,const double tau)
{
   SoftUpdateDQN(gDQN[symIdx][regime], gTargetDQN[symIdx][regime], tau);
}

void SyncAllTargetNetworks()
{
   for(int i=0;i<gSymbolCount;i++)
      for(int r=0;r<REGIME_COUNT;r++)
         SyncTargetNetFor(i,r);
}

void SoftUpdateDQN(const DQNNetwork &online, DQNNetwork &target, const double tau)
{
   double a=tau;
   if(a<0.0) a=0.0;
   if(a>1.0) a=1.0;

   target.input_dim  = online.input_dim;
   target.hidden_dim = online.hidden_dim;
   target.hidden_dim2= online.hidden_dim2;
   target.output_dim = online.output_dim;
   target.fusion_dim = online.fusion_dim;

   target.basket_h1    = online.basket_h1;
   target.basket_h2    = online.basket_h2;
   target.indicator_h1 = online.indicator_h1;
   target.indicator_h2 = online.indicator_h2;
   target.volatility_h1= online.volatility_h1;
   target.volatility_h2= online.volatility_h2;
   target.structure_h1 = online.structure_h1;
   target.structure_h2 = online.structure_h2;
   target.zone_h1      = online.zone_h1;
   target.zone_h2      = online.zone_h2;

   #define SOFT_ARR(name) ArrayResize(target.name, ArraySize(online.name)); for(int i=0;i<ArraySize(online.name);i++) target.name[i]=(1.0-a)*target.name[i] + a*online.name[i];
   SOFT_ARR(basket_W1); SOFT_ARR(basket_b1); SOFT_ARR(basket_W2); SOFT_ARR(basket_b2);
   SOFT_ARR(indicator_W1); SOFT_ARR(indicator_b1); SOFT_ARR(indicator_W2); SOFT_ARR(indicator_b2);
   SOFT_ARR(volatility_W1); SOFT_ARR(volatility_b1); SOFT_ARR(volatility_W2); SOFT_ARR(volatility_b2);
   SOFT_ARR(structure_W1); SOFT_ARR(structure_b1); SOFT_ARR(structure_W2); SOFT_ARR(structure_b2);
   SOFT_ARR(zone_W1); SOFT_ARR(zone_b1); SOFT_ARR(zone_W2); SOFT_ARR(zone_b2);
   SOFT_ARR(W1); SOFT_ARR(b1); SOFT_ARR(W2); SOFT_ARR(b2); SOFT_ARR(WV); SOFT_ARR(bV); SOFT_ARR(WA); SOFT_ARR(bA); SOFT_ARR(feat_mean); SOFT_ARR(feat_std);
   #undef SOFT_ARR
}

bool SaveDQNForSymbol(int symIdx,int regime,string filename)
{
   int handle=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(handle==INVALID_HANDLE) return false;

   int version=7;
   DQNNetwork net=gDQN[symIdx][regime];
   string metaNote=BuildCurrentDQNMetadataNote();

   FileWriteInteger(handle,version);
   FileWriteInteger(handle,net.input_dim);
   FileWriteInteger(handle,net.hidden_dim);
   FileWriteInteger(handle,net.hidden_dim2);
   FileWriteInteger(handle,net.output_dim);
   FileWriteInteger(handle,net.fusion_dim);

   FileWriteInteger(handle,net.basket_h1);    FileWriteInteger(handle,net.basket_h2);
   FileWriteInteger(handle,net.indicator_h1); FileWriteInteger(handle,net.indicator_h2);
   FileWriteInteger(handle,net.volatility_h1);FileWriteInteger(handle,net.volatility_h2);
   FileWriteInteger(handle,net.structure_h1); FileWriteInteger(handle,net.structure_h2);
   FileWriteInteger(handle,net.zone_h1);      FileWriteInteger(handle,net.zone_h2);

   WriteDoubleArray(handle, net.basket_W1);    WriteDoubleArray(handle, net.basket_b1);
   WriteDoubleArray(handle, net.basket_W2);    WriteDoubleArray(handle, net.basket_b2);
   WriteDoubleArray(handle, net.indicator_W1); WriteDoubleArray(handle, net.indicator_b1);
   WriteDoubleArray(handle, net.indicator_W2); WriteDoubleArray(handle, net.indicator_b2);
   WriteDoubleArray(handle, net.volatility_W1);WriteDoubleArray(handle, net.volatility_b1);
   WriteDoubleArray(handle, net.volatility_W2);WriteDoubleArray(handle, net.volatility_b2);
   WriteDoubleArray(handle, net.structure_W1); WriteDoubleArray(handle, net.structure_b1);
   WriteDoubleArray(handle, net.structure_W2); WriteDoubleArray(handle, net.structure_b2);
   WriteDoubleArray(handle, net.zone_W1);      WriteDoubleArray(handle, net.zone_b1);
   WriteDoubleArray(handle, net.zone_W2);      WriteDoubleArray(handle, net.zone_b2);

   WriteDoubleArray(handle, net.W1);
   WriteDoubleArray(handle, net.b1);
   WriteDoubleArray(handle, net.W2);
   WriteDoubleArray(handle, net.b2);
   WriteDoubleArray(handle, net.WV);
   WriteDoubleArray(handle, net.bV);
   WriteDoubleArray(handle, net.WA);
   WriteDoubleArray(handle, net.bA);
   WriteDoubleArray(handle, net.feat_mean);
   WriteDoubleArray(handle, net.feat_std);

   FileWriteInteger(handle, (FastTrainingMode ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipIndicatorBranch ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipVolatilityBranch ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipStructureBranch ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipZoneCandleBranch ? 1 : 0));
   FileWriteInteger(handle, MathMax(1, TrainEveryNDecisionBars));
   FileWriteString(handle, metaNote);

   FileClose(handle);
   return true;
}

bool LoadDQNForSymbol(int symIdx,int regime,string filename)
{
   if(!FileIsExist(filename)) return false;

   int handle=FileOpen(filename, FILE_READ|FILE_BIN);
   if(handle==INVALID_HANDLE) return false;

   int version=FileReadInteger(handle);
   if(version<1)
   {
      FileClose(handle);
      return false;
   }

   DQNNetwork net=gDQN[symIdx][regime];
   net.input_dim  = FileReadInteger(handle);
   net.hidden_dim = FileReadInteger(handle);
   if(version>=7)
   {
      net.hidden_dim2= FileReadInteger(handle);
      net.output_dim = FileReadInteger(handle);
   }
   else
   {
      net.hidden_dim2= 0;
      net.output_dim = FileReadInteger(handle);
   }

   if(version>=5)
   {
      net.fusion_dim = FileReadInteger(handle);

      net.basket_h1    = FileReadInteger(handle); net.basket_h2    = FileReadInteger(handle);
      net.indicator_h1 = FileReadInteger(handle); net.indicator_h2 = FileReadInteger(handle);
      net.volatility_h1= FileReadInteger(handle); net.volatility_h2= FileReadInteger(handle);
      net.structure_h1 = FileReadInteger(handle); net.structure_h2 = FileReadInteger(handle);
      net.zone_h1      = FileReadInteger(handle); net.zone_h2      = FileReadInteger(handle);

      ReadDoubleArray(handle, net.basket_W1);    ReadDoubleArray(handle, net.basket_b1);
      ReadDoubleArray(handle, net.basket_W2);    ReadDoubleArray(handle, net.basket_b2);
      ReadDoubleArray(handle, net.indicator_W1); ReadDoubleArray(handle, net.indicator_b1);
      ReadDoubleArray(handle, net.indicator_W2); ReadDoubleArray(handle, net.indicator_b2);
      ReadDoubleArray(handle, net.volatility_W1);ReadDoubleArray(handle, net.volatility_b1);
      ReadDoubleArray(handle, net.volatility_W2);ReadDoubleArray(handle, net.volatility_b2);
      ReadDoubleArray(handle, net.structure_W1); ReadDoubleArray(handle, net.structure_b1);
      ReadDoubleArray(handle, net.structure_W2); ReadDoubleArray(handle, net.structure_b2);
      ReadDoubleArray(handle, net.zone_W1);      ReadDoubleArray(handle, net.zone_b1);
      ReadDoubleArray(handle, net.zone_W2);      ReadDoubleArray(handle, net.zone_b2);

      ReadDoubleArray(handle, net.W1);
      ReadDoubleArray(handle, net.b1);
      if(version>=7)
      {
         ReadDoubleArray(handle, net.W2);
         ReadDoubleArray(handle, net.b2);
      }
      else
      {
         ArrayResize(net.W2,0);
         ArrayResize(net.b2,0);
      }
      ReadDoubleArray(handle, net.WV);
      ReadDoubleArray(handle, net.bV);
      ReadDoubleArray(handle, net.WA);
      ReadDoubleArray(handle, net.bA);
      ReadDoubleArray(handle, net.feat_mean);
      ReadDoubleArray(handle, net.feat_std);

      gLoadedDQNFastModeMeta[symIdx]=false;
      gLoadedDQNMetaNote[symIdx]="legacy_v5_model; no fast-training metadata saved";

      if(version>=6)
      {
         gLoadedDQNFastModeMeta[symIdx] = (FileReadInteger(handle)!=0);
         int skipIndicator = FileReadInteger(handle);
         int skipVolatility= FileReadInteger(handle);
         int skipStructure = FileReadInteger(handle);
         int skipZone      = FileReadInteger(handle);
         int replayEveryN  = FileReadInteger(handle);
         gLoadedDQNMetaNote[symIdx]=FileReadString(handle);
         if(StringLen(gLoadedDQNMetaNote[symIdx])<=0)
            gLoadedDQNMetaNote[symIdx]=StringFormat("fast_training=%d; skip_indicator=%d; skip_volatility=%d; skip_structure=%d; skip_zone_candle=%d; replay_every_n_decision_bars=%d",
                                                    (gLoadedDQNFastModeMeta[symIdx] ? 1 : 0),
                                                    skipIndicator,
                                                    skipVolatility,
                                                    skipStructure,
                                                    skipZone,
                                                    replayEveryN);
      }

      gDQN[symIdx][regime] = net;
      FileClose(handle);
      SyncTargetNetFor(symIdx, regime);
      return true;
   }

   FileClose(handle);
   return false;
}

void DuelingComposeQ(const double valueScalar,
                     double &adv[],
                     double &qOut[])
{
   int outDim=ArraySize(adv);
   ArrayResize(qOut,outDim);

   if(outDim<=0) return;

   double meanAdv=0.0;
   for(int o=0;o<outDim;o++)
      meanAdv += adv[o];
   meanAdv /= (double)outDim;

   for(int o=0;o<outDim;o++)
      qOut[o] = valueScalar + (adv[o] - meanAdv);
}

int ArgMaxQ(double &q[])
{
   int n=ArraySize(q);
   if(n<=0) return 0;

   int best=0;
   double bestVal=q[0];

   for(int i=1;i<n;i++)
   {
      if(q[i]>bestVal)
      {
         bestVal=q[i];
         best=i;
      }
   }
   return best;
}

double HuberLossGrad(const double error,const double delta)
{
   double a=MathAbs(error);
   if(a<=delta)
      return error;
   return (error>0.0 ? delta : -delta);
}

double ClipScalar(const double v,const double clipAbs)
{
   if(clipAbs<=0.0) return v;
   if(v> clipAbs) return clipAbs;
   if(v<-clipAbs) return -clipAbs;
   return v;
}

void ClipArrayInPlace(double &arr[],const double clipAbs)
{
   if(clipAbs<=0.0) return;
   int n=ArraySize(arr);
   for(int i=0;i<n;i++)
      arr[i]=ClipScalar(arr[i],clipAbs);
}

double ComputeBootstrappedQTarget(int symIdx,int regime,double reward,double &nextState[],bool done)
{
   if(done) return reward;

   int outDim=gDQN[symIdx][regime].output_dim;
   if(outDim<=0) return reward;

   if(UseDoubleDQN)
   {
      double qNextOnline[];
      DQNForward(symIdx,regime,nextState,qNextOnline);

      int bestNextAction=ArgMaxQ(qNextOnline);
      if(bestNextAction<0) bestNextAction=0;
      if(bestNextAction>=outDim) bestNextAction=outDim-1;

      double qNextTarget[];
      if(UseTargetNet) DQNForwardTarget(symIdx,regime,nextState,qNextTarget);
      else             DQNForward(symIdx,regime,nextState,qNextTarget);

      return reward + DQNGamma * qNextTarget[bestNextAction];
   }
   else
   {
      double qNext[];
      if(UseTargetNet) DQNForwardTarget(symIdx,regime,nextState,qNext);
      else             DQNForward(symIdx,regime,nextState,qNext);

      double maxNext=qNext[0];
      for(int o=1;o<outDim;o++)
         if(qNext[o]>maxNext) maxNext=qNext[o];

      return reward + DQNGamma * maxNext;
   }
}

double QValueForAction(int symIdx,int regime,double &state[],int action)
{
   double q[];
   DQNForwardInference(symIdx,regime,state,q);

   int outDim=ArraySize(q);
   if(outDim<=0) return 0.0;
   if(action<0) action=0;
   if(action>=outDim) action=outDim-1;

   return q[action];
}

double ComputeReplayPriorityFromTransition(int symIdx,
                                           int regime,
                                           double &state[],
                                           int action,
                                           double reward,
                                           double &nextState[],
                                           bool done)
{
   double target = ComputeBootstrappedQTarget(symIdx,regime,reward,nextState,done);
   double qNow   = QValueForAction(symIdx,regime,state,action);
   double tdErr  = target - qNow;
   return MathAbs(tdErr) + ReplayPriorityEps;
}

void DQNForwardNet(const DQNNetwork &net,const double &state[],double &qOut[])
{
   int inDim=net.input_dim;
   if(UseBranchScaffold && gBranchLayout.totalCount != inDim)
      InitBranchLayoutForStateDim(inDim);
   int fusionDim=net.fusion_dim;
   int hidDim1=net.hidden_dim;
   int hidDim2=net.hidden_dim2;
   bool useSecondTrunk = (hidDim2>0 && ArraySize(net.W2)==hidDim2*hidDim1 && ArraySize(net.b2)==hidDim2);
   int headDim=(useSecondTrunk ? hidDim2 : hidDim1);
   int outDim=net.output_dim;

   double x[];
   ArrayResize(x,inDim);
   int sz=ArraySize(state);

   for(int i=0;i<inDim;i++)
   {
      double v=(i<sz ? state[i] : 0.0);
      if(UseInputNorm && i < ArraySize(net.feat_std) && net.feat_std[i] > 0.0)
         v=(v - net.feat_mean[i]) / net.feat_std[i];
      x[i]=v;
   }

   double basket_z1[],basket_a1[],basket_z2[],basket_a2[];
   double indicator_z1[],indicator_a1[],indicator_z2[],indicator_a2[];
   double volatility_z1[],volatility_a1[],volatility_z2[],volatility_a2[];
   double structure_z1[],structure_a1[],structure_z2[],structure_a2[];
   double zone_z1[],zone_a1[],zone_z2[],zone_a2[];

   ForwardBranchEncoder(x, gBranchLayout.basketStart, gBranchLayout.basketCount,
                        net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2,
                        basket_z1, basket_a1, basket_z2, basket_a2);
   ForwardBranchEncoder(x, gBranchLayout.indicatorStart, gBranchLayout.indicatorCount,
                        net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2,
                        indicator_z1, indicator_a1, indicator_z2, indicator_a2);
   ForwardBranchEncoder(x, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount,
                        net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2,
                        volatility_z1, volatility_a1, volatility_z2, volatility_a2);
   ForwardBranchEncoder(x, gBranchLayout.structureStart, gBranchLayout.structureCount,
                        net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2,
                        structure_z1, structure_a1, structure_z2, structure_a2);
   ForwardBranchEncoder(x, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount,
                        net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2,
                        zone_z1, zone_a1, zone_z2, zone_a2);

   double fusion[];
   ArrayResize(fusion, fusionDim);
   int cursor=0;
   for(int i=0;i<ArraySize(basket_a2);i++) fusion[cursor++]=basket_a2[i];
   for(int i=0;i<ArraySize(indicator_a2);i++) fusion[cursor++]=indicator_a2[i];
   for(int i=0;i<ArraySize(volatility_a2);i++) fusion[cursor++]=volatility_a2[i];
   for(int i=0;i<ArraySize(structure_a2);i++) fusion[cursor++]=structure_a2[i];
   for(int i=0;i<ArraySize(zone_a2);i++) fusion[cursor++]=zone_a2[i];
   while(cursor<fusionDim) fusion[cursor++]=0.0;

   double h1[];
   ArrayResize(h1,hidDim1);
   for(int hh=0;hh<hidDim1;hh++)
   {
      double sum=net.b1[hh];
      for(int i=0;i<fusionDim;i++)
         sum += net.W1[W1Index(fusionDim,hh,i)] * fusion[i];
      h1[hh]=SiLU(sum);
   }

   double h2[];
   if(useSecondTrunk)
   {
      ArrayResize(h2,hidDim2);
      for(int hh=0;hh<hidDim2;hh++)
      {
         double sum=net.b2[hh];
         for(int i=0;i<hidDim1;i++)
            sum += net.W2[DenseIndex(hidDim1,hh,i)] * h1[i];
         h2[hh]=SiLU(sum);
      }
   }

   double valueScalar = net.bV[0];
   for(int hh=0;hh<headDim;hh++)
      valueScalar += net.WV[WVIndex(headDim,0,hh)] * (useSecondTrunk ? h2[hh] : h1[hh]);

   double adv[];
   ArrayResize(adv,outDim);

   for(int o=0;o<outDim;o++)
   {
      double sum=net.bA[o];
      for(int hh=0;hh<headDim;hh++)
         sum += net.WA[WAIndex(headDim,o,hh)] * (useSecondTrunk ? h2[hh] : h1[hh]);
      adv[o]=sum;
   }

   DuelingComposeQ(valueScalar,adv,qOut);
}


void DQNForward(int symIdx,int regime,const double &state[],double &qOut[])
{
   DQNForwardNet(gDQN[symIdx][regime],state,qOut);
}

void DQNForwardTarget(int symIdx,int regime,const double &state[],double &qOut[])
{
   DQNForwardNet(gTargetDQN[symIdx][regime],state,qOut);
}

void DQNUpdateSingle(int symIdx,int regime,double &state[],int action,double reward,double &nextState[],bool done)
{
   DQNNetwork net = gDQN[symIdx][regime];
   int inDim = net.input_dim;
   if(UseBranchScaffold && gBranchLayout.totalCount != inDim)
      InitBranchLayoutForStateDim(inDim);

   int fusionDim = net.fusion_dim;
   int hidDim1   = net.hidden_dim;
   int hidDim2   = net.hidden_dim2;
   bool useSecondTrunk = (hidDim2>0 && ArraySize(net.W2)==hidDim2*hidDim1 && ArraySize(net.b2)==hidDim2);
   int headDim   = (useSecondTrunk ? hidDim2 : hidDim1);
   int outDim    = net.output_dim;

   if(action<0 || action>=outDim) return;

   double x[];
   ArrayResize(x,inDim);
   int sz=ArraySize(state);

   for(int i=0;i<inDim;i++)
   {
      double v=(i<sz ? state[i] : 0.0);
      if(UseInputNorm && i < ArraySize(net.feat_std) && net.feat_std[i] > 0.0)
         v=(v - net.feat_mean[i]) / net.feat_std[i];
      x[i]=v;
   }

   double basket_z1[],basket_a1[],basket_z2[],basket_a2[];
   double indicator_z1[],indicator_a1[],indicator_z2[],indicator_a2[];
   double volatility_z1[],volatility_a1[],volatility_z2[],volatility_a2[];
   double structure_z1[],structure_a1[],structure_z2[],structure_a2[];
   double zone_z1[],zone_a1[],zone_z2[],zone_a2[];

   ForwardBranchEncoder(x, gBranchLayout.basketStart, gBranchLayout.basketCount,
                        net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2,
                        basket_z1, basket_a1, basket_z2, basket_a2);
   ForwardBranchEncoder(x, gBranchLayout.indicatorStart, gBranchLayout.indicatorCount,
                        net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2,
                        indicator_z1, indicator_a1, indicator_z2, indicator_a2);
   ForwardBranchEncoder(x, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount,
                        net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2,
                        volatility_z1, volatility_a1, volatility_z2, volatility_a2);
   ForwardBranchEncoder(x, gBranchLayout.structureStart, gBranchLayout.structureCount,
                        net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2,
                        structure_z1, structure_a1, structure_z2, structure_a2);
   ForwardBranchEncoder(x, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount,
                        net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2,
                        zone_z1, zone_a1, zone_z2, zone_a2);

   double fusion[];
   ArrayResize(fusion,fusionDim);
   int basketStart=0;
   int indicatorStart=ArraySize(basket_a2);
   int volatilityStart=indicatorStart + ArraySize(indicator_a2);
   int structureStart=volatilityStart + ArraySize(volatility_a2);
   int zoneStart=structureStart + ArraySize(structure_a2);

   int cursor=0;
   for(int i=0;i<ArraySize(basket_a2);i++) fusion[cursor++]=basket_a2[i];
   for(int i=0;i<ArraySize(indicator_a2);i++) fusion[cursor++]=indicator_a2[i];
   for(int i=0;i<ArraySize(volatility_a2);i++) fusion[cursor++]=volatility_a2[i];
   for(int i=0;i<ArraySize(structure_a2);i++) fusion[cursor++]=structure_a2[i];
   for(int i=0;i<ArraySize(zone_a2);i++) fusion[cursor++]=zone_a2[i];
   while(cursor<fusionDim) fusion[cursor++]=0.0;

   double h1[];
   double zTrunk1[];
   ArrayResize(h1,hidDim1);
   ArrayResize(zTrunk1,hidDim1);
   for(int hh=0;hh<hidDim1;hh++)
   {
      double sum=net.b1[hh];
      for(int i=0;i<fusionDim;i++)
         sum += net.W1[W1Index(fusionDim,hh,i)] * fusion[i];
      zTrunk1[hh]=sum;
      h1[hh]=SiLU(sum);
   }

   double h2[];
   double zTrunk2[];
   double finalH[];
   double finalZ[];
   if(useSecondTrunk)
   {
      ArrayResize(h2,hidDim2);
      ArrayResize(zTrunk2,hidDim2);
      for(int hh=0;hh<hidDim2;hh++)
      {
         double sum=net.b2[hh];
         for(int i=0;i<hidDim1;i++)
            sum += net.W2[DenseIndex(hidDim1,hh,i)] * h1[i];
         zTrunk2[hh]=sum;
         h2[hh]=SiLU(sum);
      }
      ArrayResize(finalH,hidDim2);
      ArrayResize(finalZ,hidDim2);
      for(int i=0;i<hidDim2;i++){ finalH[i]=h2[i]; finalZ[i]=zTrunk2[i]; }
   }
   else
   {
      ArrayResize(finalH,hidDim1);
      ArrayResize(finalZ,hidDim1);
      for(int i=0;i<hidDim1;i++){ finalH[i]=h1[i]; finalZ[i]=zTrunk1[i]; }
   }

   double valueScalar = net.bV[0];
   for(int hh=0;hh<headDim;hh++)
      valueScalar += net.WV[WVIndex(headDim,0,hh)] * finalH[hh];

   double adv[];
   ArrayResize(adv,outDim);
   for(int o=0;o<outDim;o++)
   {
      double sum=net.bA[o];
      for(int hh=0;hh<headDim;hh++)
         sum += net.WA[WAIndex(headDim,o,hh)] * finalH[hh];
      adv[o]=sum;
   }

   double q[];
   DuelingComposeQ(valueScalar,adv,q);

   double target = ComputeBootstrappedQTarget(symIdx,regime,reward,nextState,done);
   double tdError = target - q[action];

   double dLoss_dQaction;
   if(UseHuberLoss)
      dLoss_dQaction = -HuberLossGrad(tdError,HuberDelta);
   else
      dLoss_dQaction = -tdError;

   double lr = DQNLearningRate;
   if(UseDangerBrain) lr *= gLRScale[symIdx];

   double dLoss_dV = dLoss_dQaction;

   double dLoss_dAdv[];
   ArrayResize(dLoss_dAdv,outDim);
   double invOut = 1.0 / (double)outDim;

   for(int o=0;o<outDim;o++)
   {
      if(o==action) dLoss_dAdv[o] = dLoss_dQaction * (1.0 - invOut);
      else          dLoss_dAdv[o] = dLoss_dQaction * (0.0 - invOut);
   }

   if(UseGradientClipping)
   {
      dLoss_dV = ClipScalar(dLoss_dV,GradientClipValue);
      ClipArrayInPlace(dLoss_dAdv,GradientClipValue);
   }

   double oldWV[];
   ArrayResize(oldWV,headDim);
   for(int hh=0;hh<headDim;hh++) oldWV[hh] = net.WV[WVIndex(headDim,0,hh)];

   double oldWA[];
   ArrayResize(oldWA,outDim*headDim);
   for(int i=0;i<ArraySize(oldWA);i++) oldWA[i] = net.WA[i];

   for(int hh=0;hh<headDim;hh++)
   {
      int idxWV = WVIndex(headDim,0,hh);
      double grad = dLoss_dV * finalH[hh];
      if(UseGradientClipping) grad = ClipScalar(grad,GradientClipValue);
      net.WV[idxWV] -= lr * grad;
   }
   {
      double gradb = dLoss_dV;
      if(UseGradientClipping) gradb = ClipScalar(gradb,GradientClipValue);
      net.bV[0] -= lr * gradb;
   }

   for(int o=0;o<outDim;o++)
   {
      for(int hh=0;hh<headDim;hh++)
      {
         int idxWA = WAIndex(headDim,o,hh);
         double grad = dLoss_dAdv[o] * finalH[hh];
         if(UseGradientClipping) grad = ClipScalar(grad,GradientClipValue);
         net.WA[idxWA] -= lr * grad;
      }
      double gradb = dLoss_dAdv[o];
      if(UseGradientClipping) gradb = ClipScalar(gradb,GradientClipValue);
      net.bA[o] -= lr * gradb;
   }

   double dHead[];
   ArrayResize(dHead,headDim);
   for(int hh=0;hh<headDim;hh++)
   {
      double grad = oldWV[hh] * dLoss_dV;
      for(int o=0;o<outDim;o++)
         grad += oldWA[WAIndex(headDim,o,hh)] * dLoss_dAdv[o];
      grad *= SiLUDerivativeFromPreAct(finalZ[hh]);
      dHead[hh] = grad;
   }
   if(UseGradientClipping) ClipArrayInPlace(dHead,GradientClipValue);

   double dTrunk1[];
   if(useSecondTrunk)
   {
      double oldW2[];
      ArrayResize(oldW2,ArraySize(net.W2));
      for(int i=0;i<ArraySize(net.W2);i++) oldW2[i]=net.W2[i];

      for(int r=0;r<hidDim2;r++)
      {
         for(int c=0;c<hidDim1;c++)
         {
            int idx=DenseIndex(hidDim1,r,c);
            double grad=dHead[r]*h1[c];
            if(UseGradientClipping) grad=ClipScalar(grad,GradientClipValue);
            net.W2[idx] -= lr*grad;
         }
         double gradb=dHead[r];
         if(UseGradientClipping) gradb=ClipScalar(gradb,GradientClipValue);
         net.b2[r] -= lr*gradb;
      }

      ArrayResize(dTrunk1,hidDim1);
      for(int c=0;c<hidDim1;c++)
      {
         double s=0.0;
         for(int r=0;r<hidDim2;r++)
            s += oldW2[DenseIndex(hidDim1,r,c)] * dHead[r];
         dTrunk1[c]=s * SiLUDerivativeFromPreAct(zTrunk1[c]);
      }
   }
   else
   {
      ArrayResize(dTrunk1,hidDim1);
      for(int i=0;i<hidDim1;i++) dTrunk1[i]=dHead[i];
   }
   if(UseGradientClipping) ClipArrayInPlace(dTrunk1,GradientClipValue);

   double oldW1[];
   ArrayResize(oldW1,ArraySize(net.W1));
   for(int i=0;i<ArraySize(net.W1);i++) oldW1[i]=net.W1[i];

   for(int hh=0;hh<hidDim1;hh++)
   {
      for(int i=0;i<fusionDim;i++)
      {
         int idx = W1Index(fusionDim,hh,i);
         double grad = dTrunk1[hh] * fusion[i];
         if(UseGradientClipping) grad = ClipScalar(grad,GradientClipValue);
         net.W1[idx] -= lr * grad;
      }
      double gradb = dTrunk1[hh];
      if(UseGradientClipping) gradb = ClipScalar(gradb,GradientClipValue);
      net.b1[hh] -= lr * gradb;
   }

   double dFusion[];
   ArrayResize(dFusion,fusionDim);
   for(int i=0;i<fusionDim;i++)
   {
      double s=0.0;
      for(int hh=0;hh<hidDim1;hh++)
         s += oldW1[W1Index(fusionDim,hh,i)] * dTrunk1[hh];
      dFusion[i]=s;
   }
   if(UseGradientClipping) ClipArrayInPlace(dFusion,GradientClipValue);

   if(ArraySize(basket_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(basket_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[basketStart+i];
      BackpropBranchEncoder(x, gBranchLayout.basketStart, gBranchLayout.basketCount,
                            net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2,
                            basket_z1, basket_a1, basket_z2, g, lr);
   }
   if(ArraySize(indicator_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(indicator_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[indicatorStart+i];
      BackpropBranchEncoder(x, gBranchLayout.indicatorStart, gBranchLayout.indicatorCount,
                            net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2,
                            indicator_z1, indicator_a1, indicator_z2, g, lr);
   }
   if(ArraySize(volatility_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(volatility_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[volatilityStart+i];
      BackpropBranchEncoder(x, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount,
                            net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2,
                            volatility_z1, volatility_a1, volatility_z2, g, lr);
   }
   if(ArraySize(structure_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(structure_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[structureStart+i];
      BackpropBranchEncoder(x, gBranchLayout.structureStart, gBranchLayout.structureCount,
                            net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2,
                            structure_z1, structure_a1, structure_z2, g, lr);
   }
   if(ArraySize(zone_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(zone_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[zoneStart+i];
      BackpropBranchEncoder(x, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount,
                            net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2,
                            zone_z1, zone_a1, zone_z2, g, lr);
   }

   gDQN[symIdx][regime] = net;
}


void DQNUpdate(int symIdx,int activeRegime,double &state[],int action,double reward,double &nextState[],bool done)
{
   DQNUpdateSingle(symIdx,activeRegime,state,action,reward,nextState,done);

   if(ShouldTrainAllRegimesNow())
   {
      for(int r=0;r<REGIME_COUNT;r++)
      {
         if(r==activeRegime) continue;
         DQNUpdateSingle(symIdx,r,state,action,reward,nextState,done);
      }
   }

   gTargetSyncCounter++;
   if(UseTargetNet && TargetSyncFreq>0 && gTargetSyncCounter>=TargetSyncFreq)
   {
      SyncAllTargetNetworks();
      gTargetSyncCounter=0;
   }
}


void RefreshReplayBankPriority(const int src,const int idx,const ReplayItem &it,const double pr)
{
   double blended = MathMax(0.01, pr + ReplayPriorityReward(it.reward));
   if(src==REPLAY_SRC_MAIN)
   {
      if(idx>=0 && idx<ArraySize(gReplay))
         gReplay[idx].priority = MathMax(0.01, 0.5*gReplay[idx].priority + 0.5*blended);
      return;
   }
   if(src==REPLAY_SRC_RECENT)
   {
      if(idx>=0 && idx<ArraySize(gRecentReplayBank.items))
         gRecentReplayBank.items[idx].priority = MathMax(0.01, 0.5*gRecentReplayBank.items[idx].priority + 0.5*blended);
      return;
   }
   if(src==REPLAY_SRC_DANGER)
   {
      if(idx>=0 && idx<ArraySize(gDangerReplayBank.items))
         gDangerReplayBank.items[idx].priority = MathMax(0.01, (1.0-DangerBankPriorityAlpha)*gDangerReplayBank.items[idx].priority + DangerBankPriorityAlpha*(blended*(1.0+0.25*MathMax(0,it.dangerClass))));
      return;
   }
   if(src==REPLAY_SRC_DEEP)
   {
      if(idx>=0 && idx<ArraySize(gDeepBasketReplayBank.items))
      {
         double w = DeepBasketSampleWeight(it);
         gDeepBasketReplayBank.items[idx].priority = MathMax(0.01, (1.0-DeepBankPriorityAlpha)*gDeepBasketReplayBank.items[idx].priority + DeepBankPriorityAlpha*(blended * MathMax(1.0, w)));
      }
      return;
   }
   if(src==REPLAY_SRC_EFFICIENT)
   {
      if(idx>=0 && idx<ArraySize(gEfficientReplayBank.items))
      {
         double effBoost = (it.addDepthClass==0 && it.reward>0.0 ? 1.10 : 1.0);
         gEfficientReplayBank.items[idx].priority = MathMax(0.01, (1.0-EfficientBankPriorityAlpha)*gEfficientReplayBank.items[idx].priority + EfficientBankPriorityAlpha*(blended*effBoost));
      }
      return;
   }
}

void TrainReplayBatch()
{
   if(!UseReplayBuffer) return;

   int n=ArraySize(gReplay);
   if(n<ReplayWarmup || ReplayBatchSize<=0) return;

   int iters=MathMax(1,ReplayTrainIters);
   int batchN=MathMax(1,ReplayBatchSize);

   for(int t=0;t<iters;t++)
   {
      for(int b=0;b<batchN;b++)
      {
         int idx=-1;
         int sampleSrc=REPLAY_SRC_MAIN;
         ReplayItem it;
         if(!SampleReplayItemMixed(it, sampleSrc, idx)) return;
         if(sampleSrc==REPLAY_SRC_MAIN && (idx<0 || idx>=ArraySize(gReplay))) continue;
         if(it.regime<0 || it.regime>=REGIME_COUNT) continue;
         if(it.symIdx<0 || it.symIdx>=gSymbolCount) continue;

         DQNUpdateSingle(it.symIdx,it.regime,it.state,it.action,it.reward,it.nextState,it.done);

         double pr = ComputeReplayPriorityFromTransition(it.symIdx,
                                                         it.regime,
                                                         it.state,
                                                         it.action,
                                                         it.reward,
                                                         it.nextState,
                                                         it.done);

         RefreshReplayBankPriority(sampleSrc, idx, it, pr);

         if(UseTargetNet && UseSoftTargetUpdate)
            SoftUpdateTargetNetFor(it.symIdx,it.regime,SoftTargetTau);
      }
   }

   if(gReplayPendingTrainCount>0)
      gReplayPendingTrainCount--;

   if(UseTargetNet && !UseSoftTargetUpdate)
   {
      gTargetSyncCounter++;
      if(TargetSyncFreq>0 && gTargetSyncCounter>=TargetSyncFreq)
      {
         SyncAllTargetNetworks();
         gTargetSyncCounter=0;
      }
   }
}

void DDEventTrimIfNeeded()
{
   while(ArraySize(gDDEvents) > MaxDDEventsStored)
      ArrayRemove(gDDEvents,0,1);
}

void BasketHistoryPush(const BasketSnapshot &snap)
{
   int n=ArraySize(gBasketHistory);
   ArrayResize(gBasketHistory,n+1);
   gBasketHistory[n]=snap;

   if(ArraySize(gBasketHistory)>BasketHistoryCapacity)
      ArrayRemove(gBasketHistory,0,1);
}

void TickTracePush(const TickTraceItem &t)
{
   int n=ArraySize(gTickTrace);
   ArrayResize(gTickTrace,n+1);
   gTickTrace[n]=t;

   if(ArraySize(gTickTrace)>TickTraceCapacity)
      ArrayRemove(gTickTrace,0,1);
}

void CaptureBasketSnapshot(const string symbol,
                           const int symIdx,
                           const int magic,
                           const int regime,
                           const int basketDir,
                           const int positionsCount,
                           const double &state[],
                           const double &qVals[])
{
   if(!UseDDEventMemory) return;

   BasketSnapshot s;
   s.timeStamp=TimeCurrent();
   s.symbol=symbol;
   s.magic=magic;
   s.regime=regime;
   s.basketDir=basketDir;
   s.positionsCount=positionsCount;
   s.equity=GetEAEquity();
   s.openPnL=CalculatePositionsPnL(symbol,magic);

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   s.midPrice=mid;
   s.avgPrice=(positionsCount>0 && gTrades[symIdx].Total()>0 ? BasketAvgPrice(gTrades[symIdx],gTradeLots[symIdx],mid) : mid);
   s.entryPrice=s.avgPrice;
   s.gridStep=(gGridActiveStep[symIdx]>0.0 ? gGridActiveStep[symIdx] : GetGridStepCached(symIdx));
   s.danger=(UseDangerBrain ? gPDanger[symIdx] : 0.0);

   ArrayResize(s.stateKey,0);
   BuildQMemoryKey(symIdx,state,s.stateKey);

   ArrayResize(s.qVals,ArraySize(qVals));
   for(int i=0;i<ArraySize(qVals);i++) s.qVals[i]=qVals[i];

   BasketHistoryPush(s);
   
   if(gDDEventActive)
      AppendLatestSnapshotToActiveDDEvent();   
}

void AppendLatestSnapshotToActiveDDEvent()
{
   if(!UseDDEventMemory) return;
   if(!gDDEventActive) return;

   int evN=ArraySize(gDDEvents);
   int histN=ArraySize(gBasketHistory);
   if(evN<=0 || histN<=0) return;

   BasketSnapshot lastSnap = gBasketHistory[histN-1];

   int n=ArraySize(gDDEvents[evN-1].postBaskets);
   ArrayResize(gDDEvents[evN-1].postBaskets,n+1);
   gDDEvents[evN-1].postBaskets[n]=lastSnap;

   gDDEventPostBasketCount++;
}

void CaptureTickTrace(const int symIdx)
{
   if(!UseDDEventMemory) return;
   if(!gDDEventActive) return;

   string sym=gSymbols[symIdx];

   if(DDEventTraceOnBars)
   {
      ENUM_TIMEFRAMES tf=DDEventTraceTFForSymbol(sym);
      datetime barTime=iTime(sym, tf, 0);
      if(barTime<=0) return;
      if(gDDEventLastTraceBarTime[symIdx]==barTime) return;
      gDDEventLastTraceBarTime[symIdx]=barTime;
   }

   TickTraceItem t;
   t.timeStamp=TimeCurrent();

   double bid=SymbolInfoDouble(sym,SYMBOL_BID);
   double ask=SymbolInfoDouble(sym,SYMBOL_ASK);
   double point=SymbolInfoDouble(sym,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   t.bid=bid;
   t.ask=ask;
   t.mid=0.5*(bid+ask);
   t.spreadPoints=(ask-bid)/point;
   t.equity=GetEAEquity();
   t.danger=(UseDangerBrain ? gPDanger[symIdx] : 0.0);

   TickTracePush(t);
}

void StartDDEvent(const int symIdx,
                  const int regime,
                  const int basketDir,
                  const double ddMoney,
                  const bool hardTrigger,
                  const double &state[],
                  const double &qVals[])
{
   if(!UseDDEventMemory) return;
   if(gDDEventActive) return;

   gDDEventActive=true;
   gDDEventTriggerTime=TimeCurrent();
   gDDEventTriggerDD=ddMoney;
   gDDEventHard=hardTrigger;
   gDDEventStartBasketIndex=MathMax(0, ArraySize(gBasketHistory)-BasketsBeforeDD);
   gDDEventStartTickIndex=ArraySize(gTickTrace);
   gDDEventPostBasketCount=0;
   gDDEventLastTraceBarTime[symIdx]=0;

   DDEventRecord ev;
   ev.created=TimeCurrent();
   ev.triggerTime=gDDEventTriggerTime;
   ev.symbol=gSymbols[symIdx];
   ev.magic=gMagics[symIdx];
   ev.regimeAtTrigger=regime;
   ev.basketDirAtTrigger=basketDir;
   ev.ddAtTrigger=ddMoney;
   ev.hardTrigger=hardTrigger;
   ev.completed=false;

   ArrayResize(ev.triggerStateKey,0);
   BuildQMemoryKey(symIdx,state,ev.triggerStateKey);

   ArrayResize(ev.triggerQVals,ArraySize(qVals));
   for(int i=0;i<ArraySize(qVals);i++) ev.triggerQVals[i]=qVals[i];

   ArrayResize(ev.preBaskets,0);
   int fromIdx=gDDEventStartBasketIndex;
   int toIdx=ArraySize(gBasketHistory)-1;
   for(int i=fromIdx;i<=toIdx;i++)
   {
      int n=ArraySize(ev.preBaskets);
      ArrayResize(ev.preBaskets,n+1);
      ev.preBaskets[n]=gBasketHistory[i];
   }

   ArrayResize(ev.postBaskets,0);
   ArrayResize(ev.ticks,0);

   int m=ArraySize(gDDEvents);
   ArrayResize(gDDEvents,m+1);
   gDDEvents[m]=ev;

   DDEventTrimIfNeeded();
}



void FinalizeActiveDDEvent()
{
   if(!UseDDEventMemory) return;
   if(!gDDEventActive) return;

   int n=ArraySize(gDDEvents);
   if(n<=0)
   {
      gDDEventActive=false;
      gDDEventTriggerTime=0;
      gDDEventTriggerDD=0.0;
      gDDEventHard=false;
      gDDEventStartBasketIndex=-1;
      gDDEventStartTickIndex=-1;
      gDDEventPostBasketCount=0;
      return;
   }

   int tickFrom=gDDEventStartTickIndex;
   if(tickFrom<0) tickFrom=0;
   if(tickFrom>ArraySize(gTickTrace)) tickFrom=ArraySize(gTickTrace);

   int tickN=ArraySize(gTickTrace)-tickFrom;
   if(tickN<0) tickN=0;
   if(tickN>MaxTicksPerEvent) tickN=MaxTicksPerEvent;

   ArrayResize(gDDEvents[n-1].ticks,tickN);
   for(int i=0;i<tickN;i++)
      gDDEvents[n-1].ticks[i]=gTickTrace[tickFrom+i];

   gDDEvents[n-1].completed=true;

   gDDEventActive=false;
   gDDEventTriggerTime=0;
   gDDEventTriggerDD=0.0;
   gDDEventHard=false;
   gDDEventStartBasketIndex=-1;
   gDDEventStartTickIndex=-1;
   gDDEventPostBasketCount=0;
   for(int s=0;s<MAX_SYMBOLS;s++) gDDEventLastTraceBarTime[s]=0;
}

void UpdateDDEventLifecycle(const int symIdx,
                            const int regime,
                            const int basketDir,
                            const double &state[],
                            const double &qVals[])
{
   if(!UseDDEventMemory) return;

   double eq=GetEAEquity();
   double ddMoney=(maxEquity>eq ? (maxEquity-eq) : 0.0);

   if(!gDDEventActive)
   {
      if(ddMoney >= HardDDTriggerMoney)
      {
         StartDDEvent(symIdx, regime, basketDir, ddMoney, true, state, qVals);
      }
      else if(ddMoney >= SoftDDTriggerMoney)
      {
         StartDDEvent(symIdx, regime, basketDir, ddMoney, false, state, qVals);
      }
   }
   else
   {
      CaptureTickTrace(symIdx);

      if(gDDEventPostBasketCount >= BasketsAfterDD)
         FinalizeActiveDDEvent();
   }
}

bool QueryDDEventBiasWithKey(const int symIdx,
                             const int regime,
                             const int action,
                             const int basketDir,
                             const double &stateKey[],
                             double &biasOut[])
{
   ArrayResize(biasOut, ActionCount);
   for(int a=0;a<ActionCount;a++) biasOut[a]=0.0;

   if(!UseDDEventBias) return false;
   if(ArraySize(gDDEvents)<=0) return false;
   if(ArraySize(stateKey)<=0) return false;

   double sumW=0.0;
   int start=MathMax(0, ArraySize(gDDEvents)-DDEventMaxEventsScan);

   for(int i=start;i<ArraySize(gDDEvents);i++)
   {
      if(gDDEvents[i].symbol != gSymbols[symIdx]) continue;
      if(gDDEvents[i].regimeAtTrigger!=regime && UseRegimeBank) continue;
      if(ArraySize(gDDEvents[i].triggerStateKey)!=ArraySize(stateKey)) continue;

      double sim=CosSim(stateKey,gDDEvents[i].triggerStateKey);
      if(sim < DDEventMinSim) continue;

      int triggerAction = (gDDEvents[i].basketDirAtTrigger>0 ? 1 : (gDDEvents[i].basketDirAtTrigger<0 ? 2 : 0));

      double risk = Clamp(gDDEvents[i].ddAtTrigger / MathMax(HardDDTriggerMoney, 1.0), 0.0, 2.0);
      if(gDDEvents[i].hardTrigger) risk += 0.50;
      risk = Clamp(risk, 0.0, 2.5);

      double w = sim * (1.0 + 0.25 * risk);
      if(gDDEvents[i].hardTrigger) w *= 1.15;

      biasOut[0] += w * (DDEventHoldBias + 0.35 * risk);

      if(triggerAction>=1 && triggerAction<=2)
      {
         double dirPenalty = DDEventPrePenalty + DDEventExpandPenalty * risk;
         biasOut[triggerAction] -= w * dirPenalty;
         if(action==triggerAction)
            biasOut[action] -= 0.35 * w * dirPenalty;
      }

      if(ArraySize(gDDEvents[i].postBaskets)>0)
      {
         int sameDirCount=0;
         int oppDirCount=0;
         for(int b=0;b<ArraySize(gDDEvents[i].postBaskets);b++)
         {
            int dir=gDDEvents[i].postBaskets[b].basketDir;
            if(dir==gDDEvents[i].basketDirAtTrigger) sameDirCount++;
            else if(dir==-gDDEvents[i].basketDirAtTrigger) oppDirCount++;
         }

         if(triggerAction>=1 && triggerAction<=2)
         {
            if(sameDirCount>oppDirCount)
            {
               biasOut[0] += w * 0.25 * DDEventHoldBias;
               biasOut[triggerAction] -= w * DDEventSameDirPenalty;
            }
            else if(!DDEventUseCautionOnly && oppDirCount>sameDirCount)
            {
               int altAction = (triggerAction==1 ? 2 : 1);
               biasOut[altAction] += w * DDEventRecoveryBoost;
            }
         }
      }

      sumW += w;
   }

   if(sumW<=1e-12) return false;

   for(int a=0;a<ActionCount;a++)
      biasOut[a] /= sumW;

   return true;
}
int DQNSelectAction(int symIdx,int regime,double &state[])
{
   double eps=currentEpsilon;
   if(UseDangerBrain && gMode[symIdx]==MODE_DANGER) eps *= 0.5;

   if((double)MathRand()/32767.0 < eps)
      return (MathRand() % ActionCount);

   double qBase[];
   DQNForwardInference(symIdx,regime,state,qBase);

   double qAdj[];
   ArrayResize(qAdj, ArraySize(qBase));
   for(int a=0;a<ArraySize(qBase);a++) qAdj[a]=qBase[a];

   double totalDelta[];
   InitActionBias(totalDelta);

   int basketDir=0;
   if(gPositionsCount[symIdx]>0)
      basketDir=BasketDir(gSymbols[symIdx], gMagics[symIdx]);

   DecisionSupportContext support;
   BuildDecisionSupportContext(symIdx, regime, state, qBase, false, basketDir, support);
   ApplyDecisionSupportDelta(support,totalDelta);

   if(UseQualityAdaptiveEntry && gPositionsCount[symIdx]<=0)
   {
      double qualityDelta[];
      BuildAdaptiveEntryQualityDelta(symIdx, regime, qBase, support, qualityDelta);
      AccumulateActionBias(totalDelta,qualityDelta);
   }

   ApplyBudgetedSubordinateBias(symIdx, false, qBase, totalDelta, qAdj);

   string symbol=gSymbols[symIdx];
   ApplyFlexibleDirectionalDiscipline(symbol, symIdx, regime, support, false, qAdj);
   ApplyConfirmedSetupAccelerator(symbol, symIdx, regime, support, qAdj);

   int best=0;
   double maxQ=qAdj[0];
   for(int a=1;a<ActionCount;a++)
      if(qAdj[a]>maxQ){ maxQ=qAdj[a]; best=a; }

   return best;
}


int TrendDirFromH1(const string symbol, const int symIdx)
{
   if(!UseH1Features) return 0;

   double atrS = gATRslow_BaseVal[symIdx];
   if(atrS<=1e-12) atrS=1e-12;

   double close1 = iClose(symbol, H1_TF, 1);
   double ema1   = gEMA_H1v[symIdx];

   double z = (close1 - ema1) / atrS;
   if(z >  0.15) return +1;
   if(z < -0.15) return -1;
   return 0;
}

int BasketDir(const string symbol, const int magic)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx])
   {
      if(gPositionsCount[idx] <= 0) return 0;
      if(gHasPositionTypeCache[idx]) return gBasketDirCache[idx];
   }

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      return (pt==POSITION_TYPE_BUY ? +1 : -1);
   }

   return 0;
}

void BuildBadEpisodeBiasSmart(const int label, double &bias[], const double &qSnap[], const int qN)
{
   ArrayResize(bias, ActionCount);
   for(int a=0;a<ActionCount;a++) bias[a]=0.0;

   if(ActionCount>=1) bias[0] += BadBias_HoldBoost;

   if(label==LBL_AGAINST_UPTREND)
   {
      if(ActionCount>=2) bias[1] += BadBias_WithTrendBoost;
      if(ActionCount>=3) bias[2] -= BadBias_AgainstPenalty;
   }
   else
   {
      if(ActionCount>=2) bias[1] -= BadBias_AgainstPenalty;
      if(ActionCount>=3) bias[2] += BadBias_WithTrendBoost;
   }

   if(qN>0 && qN==ActionCount)
   {
      int best=0;
      double mx=qSnap[0];
      for(int a=1;a<ActionCount;a++) if(qSnap[a]>mx){ mx=qSnap[a]; best=a; }
      bias[best] -= BadBias_RepeatPenalty;
   }
}
void MergeProto(ProtoEntry &p,
                const double &f[], const double &miniMaybe[], const bool haveMini,
                const double &dfpMaybe[], const bool haveDFP,
                const double &dminiMaybe[], const bool haveDMini,
                const double &bias[], const double fAlpha, const double bAlpha)
{
   int nF=ArraySize(p.features);
   for(int i=0;i<nF;i++)
      p.features[i] = (1.0-fAlpha)*p.features[i] + fAlpha*f[i];
   NormalizeVec(p.features);

   if(haveMini && UseMiniStateFingerprint && ArraySize(p.stateMini)==MINI_DIM && ArraySize(miniMaybe)==MINI_DIM)
   {
      for(int k=0;k<MINI_DIM;k++)
         p.stateMini[k] = (1.0-fAlpha)*p.stateMini[k] + fAlpha*miniMaybe[k];
      NormalizeVecN(p.stateMini);
   }

   if(haveDFP && UseDeltaSimilarity && ArraySize(p.deltaFP)==6 && ArraySize(dfpMaybe)==6)
   {
      for(int k=0;k<6;k++)
         p.deltaFP[k] = (1.0-fAlpha)*p.deltaFP[k] + fAlpha*dfpMaybe[k];
      NormalizeVec(p.deltaFP);
   }

   if(haveDMini && UseDeltaSimilarity && ArraySize(p.deltaMini)==MINI_DIM && ArraySize(dminiMaybe)==MINI_DIM)
   {
      for(int k=0;k<MINI_DIM;k++)
         p.deltaMini[k] = (1.0-fAlpha)*p.deltaMini[k] + fAlpha*dminiMaybe[k];
      NormalizeVecN(p.deltaMini);
   }

   int nB=MathMin(ArraySize(p.adapterBias), ArraySize(bias));
   for(int a=0;a<nB;a++)
      p.adapterBias[a] = (1.0-bAlpha)*p.adapterBias[a] + bAlpha*bias[a];

   p.survivalScore += 1.0;
   p.lastUsed = TimeCurrent();
}

int FindMergeCandidate(const bool wantDanger,
                       const int label, const int basketDir, const int trendDir,
                       const int symIdx,
                       const double &f_now[], const double &mini_now[], const bool haveMini,
                       const double &dfp_now[], const bool haveDFP,
                       const double &dmini_now[], const bool haveDMini,
                       double &bestSimOut)
{
   bestSimOut=-1e9;
   int best=-1;

   int n=ArraySize(gProtos);
   for(int i=0;i<n;i++)
   {
      if(gProtos[i].isDanger != wantDanger) continue;
      if(wantDanger && gProtos[i].label != label) continue;
      if(ArraySize(gProtos[i].features)!=ArraySize(f_now)) continue;

      double sim = CombinedSim(symIdx, f_now, mini_now, dfp_now, haveDFP, dmini_now, haveDMini, gProtos[i]);

      if(basketDir!=0 && gProtos[i].basketDir!=0 && gProtos[i].basketDir!=basketDir) continue;
      if(trendDir!=0  && gProtos[i].trendDir!=0  && gProtos[i].trendDir !=trendDir)  continue;

      if(sim>bestSimOut){ bestSimOut=sim; best=i; }
   }
   return best;
}

int LearnBadEpisodeSmart(const int symIdx, const bool missedDangerAtConfirm)
{
   if(gBadLabel[symIdx] < 0) return -1;

   double f_now[]; ArrayResize(f_now,6);
   if(gBadHaveBasketFP[symIdx])       for(int k=0;k<6;k++) f_now[k]=gBadFPBasketOpen[symIdx][k];
   else if(gBadHavePreFP[symIdx])     for(int k=0;k<6;k++) f_now[k]=gBadFPPre[symIdx][k];
   else                               for(int k=0;k<6;k++) f_now[k]=gBadFPStart[symIdx][k];
   NormalizeVec(f_now);

   double mini_now[]; ArrayResize(mini_now, MINI_DIM);
   bool haveMini=false;
   if(UseMiniStateFingerprint)
   {
      if(gBadHaveMiniBasket[symIdx]) { for(int k=0;k<MINI_DIM;k++) mini_now[k]=gBadMiniBasketOpen[symIdx][k]; haveMini=true; }
      else if(gBadHaveMiniPre[symIdx]) { for(int k=0;k<MINI_DIM;k++) mini_now[k]=gBadMiniPre[symIdx][k]; haveMini=true; }
      else if(gBadHaveMiniStart[symIdx]) { for(int k=0;k<MINI_DIM;k++) mini_now[k]=gBadMiniStart[symIdx][k]; haveMini=true; }
      if(haveMini) NormalizeVecN(mini_now);
   }

   double dfp_now[]; bool haveDFP=false;
   if(UseDeltaSimilarity && gBadHavePreFP[symIdx])
   {
      double cur[]; ArrayResize(cur,6);
      double prev[]; ArrayResize(prev,6);
      for(int k=0;k<6;k++){ cur[k]=gBadFPStart[symIdx][k]; prev[k]=gBadFPPre[symIdx][k]; }
      haveDFP = BuildDeltaVec(cur, prev, 6, dfp_now);
   }

   double dmini_now[]; bool haveDMini=false;
   if(UseDeltaSimilarity && UseMiniStateFingerprint && gBadHaveMiniPre[symIdx] && gBadHaveMiniStart[symIdx])
   {
      double cur[]; ArrayResize(cur,MINI_DIM);
      double prev[]; ArrayResize(prev,MINI_DIM);
      for(int k=0;k<MINI_DIM;k++){ cur[k]=gBadMiniStart[symIdx][k]; prev[k]=gBadMiniPre[symIdx][k]; }
      haveDMini = BuildDeltaVec(cur, prev, MINI_DIM, dmini_now);
   }

   double qSnap[]; ArrayResize(qSnap, ActionCount);
   for(int a=0;a<ActionCount;a++) qSnap[a]=0.0;
   bool haveQ=(gBadHaveQStart[symIdx] && ActionCount<=8);
   if(haveQ) for(int a=0;a<ActionCount;a++) qSnap[a]=gBadQStart[symIdx][a];

   double bias[];
   if(haveQ) BuildBadEpisodeBiasSmart(gBadLabel[symIdx], bias, qSnap, ActionCount);
   else { double emptyQ[]; ArrayResize(emptyQ,0); BuildBadEpisodeBiasSmart(gBadLabel[symIdx], bias, emptyQ, 0); }

   double bestSim=-1e9;
   int best=FindMergeCandidate(true, gBadLabel[symIdx], gBadBasketDir[symIdx], gBadTrendDir[symIdx], symIdx,
                               f_now, mini_now, haveMini, dfp_now, haveDFP, dmini_now, haveDMini, bestSim);

   if(best>=0 && bestSim>=BadProtoMergeSim)
   {
      if(UseDeltaSimilarity)
      {
         if(ArraySize(gProtos[best].deltaFP)==0 && haveDFP){ ArrayResize(gProtos[best].deltaFP,6); for(int k=0;k<6;k++) gProtos[best].deltaFP[k]=dfp_now[k]; }
         if(ArraySize(gProtos[best].deltaMini)==0 && haveDMini){ ArrayResize(gProtos[best].deltaMini,MINI_DIM); for(int k=0;k<MINI_DIM;k++) gProtos[best].deltaMini[k]=dmini_now[k]; }
      }

      MergeProto(gProtos[best], f_now, mini_now, haveMini, dfp_now, haveDFP, dmini_now, haveDMini,
                 bias, BadProtoFeatureEMA, BadProtoBiasEMA);

      ProtoScoreBump(best, missedDangerAtConfirm ? 2.0 : 1.0);
      return best;
   }

   ProtoEntry p;
   p.label     = gBadLabel[symIdx];
   p.basketDir = gBadBasketDir[symIdx];
   p.trendDir  = gBadTrendDir[symIdx];
   p.atrRatio  = gBadAtrRatio[symIdx];
   p.created   = TimeCurrent();
   p.isDanger=true;

   ArrayResize(p.features,6);
   for(int k=0;k<6;k++) p.features[k]=f_now[k];
   NormalizeVec(p.features);

   ArrayResize(p.stateMini, (haveMini ? MINI_DIM : 0));
   if(haveMini) for(int k=0;k<MINI_DIM;k++) p.stateMini[k]=mini_now[k];

   ArrayResize(p.deltaFP, (haveDFP ? 6 : 0));
   if(haveDFP) for(int k=0;k<6;k++) p.deltaFP[k]=dfp_now[k];

   ArrayResize(p.deltaMini, (haveDMini ? MINI_DIM : 0));
   if(haveDMini) for(int k=0;k<MINI_DIM;k++) p.deltaMini[k]=dmini_now[k];

   ArrayResize(p.adapterBias, ActionCount);
   for(int a=0;a<ActionCount;a++) p.adapterBias[a]=bias[a];

   ArrayResize(p.qSnap, (haveQ ? ActionCount : 0));
   if(haveQ) for(int a=0;a<ActionCount;a++) p.qSnap[a]=qSnap[a];

   p.survivalScore=1.0 + (missedDangerAtConfirm ? 2.0 : 1.0);
   p.usedCount=0;
   p.lastUsed=TimeCurrent();

   int n=ArraySize(gProtos);
   ArrayResize(gProtos,n+1);
   gProtos[n]=p;

   PruneProtosIfNeeded();
   return n;
}

void PendingInit()
{
   ArrayResize(gPending, MaxPendingTransitions);
   for(int i=0;i<ArraySize(gPending);i++)
   {
      gPending[i].active=false;
      gPending[i].symIdx=-1;
      gPending[i].regime=0;
      gPending[i].action=0;
      gPending[i].created=0;
      gPending[i].basketDir=0;
      gPending[i].positionsAtOpen=0;
      gPending[i].episodeId=0;
      ArrayResize(gPending[i].state,0);
   }
}

int PendingFindFreeSlot()
{
   for(int i=0;i<ArraySize(gPending);i++)
      if(!gPending[i].active) return i;

   int oldest=0;
   datetime t=LONG_MAX;
   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(gPending[i].created < t)
      {
         t=gPending[i].created;
         oldest=i;
      }
   }
   return oldest;
}


double GetLiveBasketAvgPrice(const string symbol,const int magic,const double fallback)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx])
   {
      if(gBasketAvgValid[idx])
         return gBasketAvgPriceCache[idx];

      if(gTrades[idx].Total()>0 && gTradeLots[idx].Total()==gTrades[idx].Total())
         return BasketAvgPrice(gTrades[idx],gTradeLots[idx],fallback);
   }

   double weighted=0.0;
   double totalLots=0.0;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      double price=PositionGetDouble(POSITION_PRICE_OPEN);
      double vol=PositionGetDouble(POSITION_VOLUME);
      weighted += price*vol;
      totalLots += vol;
   }

   if(totalLots<=0.0) return fallback;
   return weighted/totalLots;
}

bool FindLatestOpenedPositionMeta(const string symbol,
                                  const int magic,
                                  const ENUM_ORDER_TYPE orderType,
                                  const double expectedVolume,
                                  double &entryPrice,
                                  double &entryVolume,
                                  datetime &entryTime)
{
   int desiredType=(orderType==ORDER_TYPE_BUY ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);

   bool found=false;
   datetime bestTime=0;
   double bestVolDiff=DBL_MAX;
   double bestPrice=0.0;
   double bestVolume=0.0;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      if((int)PositionGetInteger(POSITION_TYPE)!=desiredType) continue;

      datetime t=(datetime)PositionGetInteger(POSITION_TIME);
      double vol=PositionGetDouble(POSITION_VOLUME);
      double px =PositionGetDouble(POSITION_PRICE_OPEN);
      double diff=MathAbs(vol-expectedVolume);

      if(!found || t>bestTime || (t==bestTime && diff<bestVolDiff))
      {
         found=true;
         bestTime=t;
         bestVolDiff=diff;
         bestPrice=px;
         bestVolume=vol;
      }
   }

   if(!found) return false;

   entryPrice=bestPrice;
   entryVolume=bestVolume;
   entryTime=bestTime;
   return true;
}

void PendingAdd(const int symIdx,
                const int regime,
                const int action,
                const int basketDir,
                const int positionsAtOpen,
                const double &state[],
                const double entryPrice,
                const double entryVolume,
                const int legIndex,
                const double basketAvgAtEntry,
                const datetime entryTime)
{
   if(!UsePendingTransitions) return;

   int idx=PendingFindFreeSlot();
   if(idx<0) return;

   gPending[idx].active=true;
   gPending[idx].symIdx=symIdx;
   gPending[idx].regime=regime;
   gPending[idx].action=action;
   gPending[idx].created=TimeCurrent();
   gPending[idx].basketDir=basketDir;
   gPending[idx].positionsAtOpen=positionsAtOpen;
   gPending[idx].legIndex=legIndex;
   gPending[idx].entryTime=entryTime;
   gPending[idx].closeTime=0;
   gPending[idx].entryPrice=entryPrice;
   gPending[idx].closePrice=0.0;
   gPending[idx].entryVolume=entryVolume;
   gPending[idx].basketAvgAtEntry=basketAvgAtEntry;
   gPending[idx].individualPnL=0.0;
   gPending[idx].denseRewardAccum=0.0;
   gPending[idx].episodeId=EnsureActiveBasketEpisode(symIdx, basketDir, MathMax(positionsAtOpen,1), entryTime);

   ArrayResize(gPending[idx].state,ArraySize(state));
   for(int i=0;i<ArraySize(state);i++) gPending[idx].state[i]=state[i];
}

bool BuildPostTradeSnapshot(const string symbol,
                            const int symIdx,
                            const int magic,
                            const ENUM_ORDER_TYPE expectedOrderType,
                            const double expectedVolume,
                            int &positionsAfter,
                            int &basketDirAfter,
                            double &entryPrice,
                            double &entryVolume,
                            double &basketAvgAfter,
                            datetime &entryTime)
{
   CountOpenPositions();

   positionsAfter = gPositionsCount[symIdx];
   basketDirAfter = BasketDir(symbol,magic);

   entryPrice  = 0.0;
   entryVolume = expectedVolume;
   entryTime   = TimeCurrent();

   bool found = FindLatestOpenedPositionMeta(symbol,
                                             magic,
                                             expectedOrderType,
                                             expectedVolume,
                                             entryPrice,
                                             entryVolume,
                                             entryTime);

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   if(!found)
   {
      entryPrice = (expectedOrderType==ORDER_TYPE_BUY ? ask : bid);
      entryVolume = expectedVolume;
      entryTime = TimeCurrent();
   }

   basketAvgAfter = GetLiveBasketAvgPrice(symbol,magic,(entryPrice>0.0 ? entryPrice : mid));

   return (positionsAfter > 0 && basketDirAfter != 0);
}

void BuildCloseNextState(const string symbol,
                         const int symIdx,
                         const int positionsAfterClose,
                         double &nextState[])
{
   CArrayDouble emptyTrades;
   BuildState(symbol,symIdx,positionsAfterClose,emptyTrades,nextState);

   bool extremeAfter=IsExtremeState(symIdx,positionsAfterClose);
   if(SeparateExtremeStates)
   {
      int sz=ArraySize(nextState);
      if(sz>0) nextState[sz-1]=(extremeAfter?1.0:0.0);
   }
}

void ArchiveClosedLegLearning(const PendingTransition &pt,const double resolvedReward)
{
   int n=ArraySize(gClosedLegHistory);
   ArrayResize(gClosedLegHistory,n+1);

   gClosedLegHistory[n].entryTime=pt.entryTime;
   gClosedLegHistory[n].closeTime=pt.closeTime;
   gClosedLegHistory[n].symbol=gSymbols[pt.symIdx];
   gClosedLegHistory[n].regime=pt.regime;
   gClosedLegHistory[n].action=pt.action;
   gClosedLegHistory[n].basketDir=pt.basketDir;
   gClosedLegHistory[n].legIndex=pt.legIndex;
   gClosedLegHistory[n].entryPrice=pt.entryPrice;
   gClosedLegHistory[n].closePrice=pt.closePrice;
   gClosedLegHistory[n].entryVolume=pt.entryVolume;
   gClosedLegHistory[n].basketAvgAtEntry=pt.basketAvgAtEntry;
   gClosedLegHistory[n].individualPnL=pt.individualPnL;
   gClosedLegHistory[n].resolvedReward=resolvedReward;

   if(MaxClosedLegHistory>0 && ArraySize(gClosedLegHistory)>MaxClosedLegHistory)
   {
      int keep=MaxClosedLegHistory;
      int drop=ArraySize(gClosedLegHistory)-keep;
      for(int i=0;i<keep;i++)
         gClosedLegHistory[i]=gClosedLegHistory[i+drop];
      ArrayResize(gClosedLegHistory,keep);
   }
}

double ComputePerPositionCloseAdjustment(const PendingTransition &pt)
{
   if(!UsePerPositionCloseLearning) return 0.0;

   double point=SymbolInfoDouble(gSymbols[pt.symIdx],SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double vol=MathMax(pt.entryVolume,1e-6);
   double pnlPerLot=pt.individualPnL/vol;
   double distPoints=0.0;
   if(pt.basketAvgAtEntry>0.0)
      distPoints=MathAbs(pt.entryPrice-pt.basketAvgAtEntry)/point;

   double holdHours=0.0;
   if(pt.closeTime>pt.entryTime)
      holdHours=(double)(pt.closeTime-pt.entryTime)/3600.0;

   double pnlAdj = Clamp(pnlPerLot * LegPnLPerLotScale, -1.5, 1.5);
   double distAdj= Clamp(distPoints * LegDistancePointsScale * (pt.individualPnL>=0.0 ? 1.0 : -1.0), -0.5, 0.5);
   double legAdj = Clamp((double)MathMax(0,pt.legIndex-1) * LegIndexRewardScale * (pt.individualPnL>=0.0 ? 1.0 : -1.0), -0.5, 0.5);
   double holdAdj= Clamp(holdHours * LegHoldHoursPenaltyScale, 0.0, 0.5);

   return pnlAdj + distAdj + legAdj - holdAdj;
}

int PendingFindBestMatch(const int symIdx,const PositionCloseItem &item,const bool &matched[])
{
   double point=SymbolInfoDouble(gSymbols[symIdx],SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   int best=-1;
   double bestScore=DBL_MAX;

   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(matched[i]) continue;
      if(!gPending[i].active) continue;
      if(gPending[i].symIdx!=symIdx) continue;
      if(gPending[i].basketDir!=item.basketDir) continue;

      double score=0.0;
      score += MathAbs(gPending[i].entryVolume-item.volume);
      score += MathAbs(gPending[i].entryPrice-item.entryPrice)/point;
      score += 0.25*MathAbs((double)(gPending[i].legIndex-item.legIndex));
      score += 0.001*MathAbs((double)(gPending[i].entryTime-item.entryTime));

      if(score<bestScore)
      {
         bestScore=score;
         best=i;
      }
   }

   return best;
}


double PendingResolveIndexCore(const int idx,
                               const double reward,
                               const bool done,
                               const double &nextState[])
{
   if(idx<0 || idx>=ArraySize(gPending)) return 0.0;
   if(!gPending[idx].active) return 0.0;

   int symIdx=gPending[idx].symIdx;
   int regime=gPending[idx].regime;
   int action=gPending[idx].action;

   double resolvedReward=reward + gPending[idx].denseRewardAccum + ComputePerPositionCloseAdjustment(gPending[idx]);

   ReplayPush(symIdx, regime, gPending[idx].state, action, resolvedReward, nextState, done);
   UpdateQMemoryResolved(symIdx, regime, gPending[idx].state, action, resolvedReward);

   if(!UseReplayBuffer)
   {
      double st[];
      ArrayResize(st,ArraySize(gPending[idx].state));
      for(int i=0;i<ArraySize(gPending[idx].state);i++) st[i]=gPending[idx].state[i];

      double ns[];
      ArrayResize(ns,ArraySize(nextState));
      for(int i=0;i<ArraySize(nextState);i++) ns[i]=nextState[i];

      DQNUpdate(symIdx, regime, st, action, resolvedReward, ns, done);
   }

   ArchiveClosedLegLearning(gPending[idx],resolvedReward);

   if(gPending[idx].episodeId>0 && (done || gPositionsCount[symIdx]<=1))
      FinalizeActiveBasketEpisode(symIdx, resolvedReward);

   gPending[idx].active=false;
   gPending[idx].episodeId=0;
   gPending[idx].denseRewardAccum=0.0;
   ArrayResize(gPending[idx].state,0);
   return resolvedReward;
}

double PendingResolveIndex(const int idx,
                           const double reward,
                           const bool done,
                           const double &nextState[])
{
   return PendingResolveIndexCore(idx,reward,done,nextState);
}

double PendingResolveIndexDetailed(const int idx,
                                   const double reward,
                                   const bool done,
                                   const double &nextState[],
                                   const PositionCloseItem &legItem)
{
   if(idx<0 || idx>=ArraySize(gPending)) return 0.0;
   if(!gPending[idx].active) return 0.0;

   gPending[idx].closeTime=legItem.closeTime;
   gPending[idx].closePrice=legItem.closePrice;
   gPending[idx].individualPnL=legItem.profit;

   return PendingResolveIndexCore(idx,reward,done,nextState);
}

int PendingResolveForSymbolClose(const int symIdx,
                                 const double reward,
                                 const bool done,
                                 const double &nextState[],
                                 PositionCloseItem &closedItems[],
                                 double &resolvedRewardSum)
{
   int resolved=0;
   resolvedRewardSum=0.0;

   bool matched[];
   ArrayResize(matched,ArraySize(gPending));
   for(int i=0;i<ArraySize(matched);i++) matched[i]=false;

   for(int j=0;j<ArraySize(closedItems);j++)
   {
      int idx=PendingFindBestMatch(symIdx,closedItems[j],matched);
      if(idx<0) continue;

      matched[idx]=true;
      double rr=PendingResolveIndexDetailed(idx, reward, done, nextState, closedItems[j]);
      resolvedRewardSum += rr;
      resolved++;
   }

   return resolved;
}

void PendingExpireOld(const datetime now)
{
   if(!UsePendingTransitions) return;
   int maxSec=MathMax(60, PendingExpireMinutes*60);

   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(!gPending[i].active) continue;
      if((now - gPending[i].created) < maxSec) continue;

      double nextState[];
      ArrayResize(nextState, ArraySize(gPending[i].state));
      for(int k=0;k<ArraySize(nextState);k++) nextState[k]=gPending[i].state[k];

      PendingResolveIndex(i, 0.0, false, nextState);
   }
}

ENUM_POSITION_TYPE GetPositionType(string symbol,int magic)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx] && gHasPositionTypeCache[idx])
      return (gBasketDirCache[idx] > 0 ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)==symbol &&
         (int)PositionGetInteger(POSITION_MAGIC)==magic)
      {
         ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         return pt;
      }
   }

   return POSITION_TYPE_BUY;
}

bool OpenPosition(string symbol,ENUM_ORDER_TYPE orderType,double volume,int stopLossPoints,int takeProfitPoints,int magic)
{
   trade.SetExpertMagicNumber(magic);
   trade.SetDeviationInPoints(Slippage);

   double vol=NormalizeVolume(symbol,volume);
   if(vol<=0.0) return false;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double price=0.0, sl=0.0, tp=0.0;

   if(orderType==ORDER_TYPE_BUY)
   {
      price=SymbolInfoDouble(symbol,SYMBOL_ASK);
      if(stopLossPoints>0) sl=price - stopLossPoints*point;
      if(takeProfitPoints>0) tp=price + takeProfitPoints*point;
   }
   else
   {
      price=SymbolInfoDouble(symbol,SYMBOL_BID);
      if(stopLossPoints>0) sl=price + stopLossPoints*point;
      if(takeProfitPoints>0) tp=price - takeProfitPoints*point;
   }

   return trade.PositionOpen(symbol,orderType,vol,price,sl,tp);
}


void SortPositionCloseItems(PositionCloseItem &items[])
{
   int n=ArraySize(items);
   for(int i=0;i<n-1;i++)
   {
      for(int j=i+1;j<n;j++)
      {
         if(items[j].entryTime < items[i].entryTime)
         {
            PositionCloseItem tmp=items[i];
            items[i]=items[j];
            items[j]=tmp;
         }
      }
   }

   for(int k=0;k<n;k++)
      items[k].legIndex=k+1;
}

int CollectBasketPositionSnapshots(const string symbol,const int magic,PositionCloseItem &items[])
{
   ArrayResize(items,0);

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      int n=ArraySize(items);
      ArrayResize(items,n+1);

      items[n].ticket=ticket;
      items[n].basketDir=((int)PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY ? +1 : -1);
      items[n].legIndex=0;
      items[n].entryTime=(datetime)PositionGetInteger(POSITION_TIME);
      items[n].closeTime=0;
      items[n].entryPrice=PositionGetDouble(POSITION_PRICE_OPEN);
      items[n].closePrice=0.0;
      items[n].volume=PositionGetDouble(POSITION_VOLUME);
      items[n].profit=PositionGetDouble(POSITION_PROFIT);
   }

   SortPositionCloseItems(items);
   return ArraySize(items);
}

bool ClosePositionsDetailed(const string symbol,
                            const int magic,
                            BasketCloseResult &result,
                            PositionCloseItem &closedItems[])
{
   ArrayResize(closedItems,0);

   result.attempted=0;
   result.closed=0;
   result.attemptedVolume=0.0;
   result.closedVolume=0.0;
   result.attemptedProfit=0.0;
   result.closedProfit=0.0;
   result.wins=0;
   result.losses=0;
   result.total=0;
   result.allClosed=false;

   PositionCloseItem openItems[];
   int count=CollectBasketPositionSnapshots(symbol,magic,openItems);
   if(count<=0) return false;

   result.attempted=count;
   result.total=count;

   for(int i=0;i<count;i++)
   {
      result.attemptedVolume += openItems[i].volume;
      result.attemptedProfit += openItems[i].profit;

      if(openItems[i].profit>0.0) result.wins++;
      else                        result.losses++;
   }

   for(int i=0;i<count;i++)
   {
      double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
      double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
      openItems[i].closeTime=TimeCurrent();
      openItems[i].closePrice=(openItems[i].basketDir>0 ? bid : ask);

      if(trade.PositionClose(openItems[i].ticket))
      {
         int n=ArraySize(closedItems);
         ArrayResize(closedItems,n+1);
         closedItems[n]=openItems[i];

         result.closed++;
         result.closedVolume += openItems[i].volume;
         result.closedProfit += openItems[i].profit;
      }
   }

   result.allClosed=(result.closed==result.attempted && result.attempted>0);

   if(result.closed>0)
   {
      gEAClosedProfit += result.closedProfit;
      gProfitCycleClosedProfit += result.closedProfit;

      if(result.allClosed)
         ArmDeepBasketStrongFirstTrade(symbol, result.closed);
   }

   return (result.closed>0);
}

bool ClosePositions(string symbol,int magic,double &closedProfitOut,int &closedCountOut)
{
   BasketCloseResult closeRes;
   PositionCloseItem closedItems[];
   bool success=ClosePositionsDetailed(symbol,magic,closeRes,closedItems);

   closedProfitOut=closeRes.closedProfit;
   closedCountOut=closeRes.closed;

   if(success && closeRes.allClosed && closeRes.closed>0)
   {
      MarkBasketClosedForCooldown(symbol);
   }

   return success;
}

void CloseAllPositions()
{
   if(UseGlobalAccountWatchdog)
   {
      for(int i=PositionsTotal()-1; i>=0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         bool shouldClose = false;

         if(WatchAllAccountPositions)
         {
            shouldClose = true;
         }
         else
         {
            int mg = (int)PositionGetInteger(POSITION_MAGIC);
            if(mg >= WatchdogMagicMin && mg <= WatchdogMagicMax)
               shouldClose = true;
         }

         if(shouldClose)
            trade.PositionClose(ticket);
      }

      for(int j=0; j<gSymbolCount; j++)
         gFirstTradeTime[j]=0;

      return;
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      double cp=0.0;
      int cc=0;
      ClosePositions(gSymbols[i],gMagics[i],cp,cc);
      gFirstTradeTime[i]=0;
   }
}

double CalculatePositionsPnL(string symbol,int magic)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx])
      return gCachedSymbolPnL[idx];

   double total=0.0;
   int n=PositionsTotal();
   for(int i=0;i<n;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)==symbol &&
         (int)PositionGetInteger(POSITION_MAGIC)==magic)
      {
         total += PositionGetDouble(POSITION_PROFIT);
      }
   }
   return total;
}

double CalculateLot(const string symbol,const int positionsCount)
{
   double scale=GetEquityBudgetScale();
   double lot = Lots * scale * MathPow(LotExponent, positionsCount);
   return NormalizeVolume(symbol, lot);
}

void CalculateLevelsSimple(const string symbol,
                           const int symIdx,
                           const int positionsCount,
                           CArrayDouble &trades,
                           const int magic,
                           double &buyLevel,
                           double &sellLevel)
{
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   double baseStep=GetGridStepCached(symIdx);
   if(baseStep<=0.0)
   {
      double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
      if(point<=0.0) point=0.00001;
      baseStep=(double)DefaultPips*point;
      if(baseStep<=0.0) baseStep=point*10.0;
   }

   double step=ComputeAdaptiveGridStep(symbol,symIdx,positionsCount,baseStep);

   gGridActiveChannel[symIdx]=0;
   gGridActiveStep[symIdx]=step;

   if(positionsCount > 0 && trades.Total() > 0)
   {
      double avg=(gBasketAvgValid[symIdx]
                  ? gBasketAvgPriceCache[symIdx]
                  : BasketAvgPrice(trades,gTradeLots[symIdx],mid));

      ENUM_POSITION_TYPE pt=GetPositionType(symbol, magic);

      if(pt==POSITION_TYPE_BUY)
      {
         buyLevel  = avg - step;
         sellLevel = 0.0;
      }
      else
      {
         sellLevel = avg + step;
         buyLevel  = 0.0;
      }
   }
   else
   {
      buyLevel  = bid - step;
      sellLevel = ask + step;
   }
}

bool CheckCCIExit()
{
   if(!UseCCI) return false;
   bool acted=false;

   for(int i=0;i<gSymbolCount;i++)
   {
      if(gPositionsCount[i]<=0) continue;

      string sym=gSymbols[i];
      int magic=gMagics[i];
      double cciVal=gCCI_Base[i];

      ENUM_POSITION_TYPE pt=GetPositionType(sym,magic);
      if((pt==POSITION_TYPE_BUY && cciVal < -CCI_Level) ||
         (pt==POSITION_TYPE_SELL && cciVal >  CCI_Level))
      {
         double cp=0.0; int cc=0;
         ClosePositions(sym,magic,cp,cc);
         acted=true;
      }
   }
   return acted;
}

double BasketBaseLotsEquivalent(const int positionsCount)
{
   return MathMax((double)MathMax(1,positionsCount) * Lots, 1e-6);
}

double BasketTotalLots(const int symIdx)
{
   double total=0.0;
   for(int i=0;i<gTradeLots[symIdx].Total();i++)
      total += gTradeLots[symIdx].At(i);
   return MathMax(total, 1e-6);
}

double ComputeHybridTakeProfitDistance(const int symIdx,const int positionsCount,const double channelStep)
{
   double point=SymbolInfoDouble(gSymbols[symIdx],SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double distChannel=MathMax(point, ChannelCloseAlpha * MathMax(channelStep, point));
   if(!UseHybridBasketTP)
      return distChannel;

   double totalLots=BasketTotalLots(symIdx);
   double legacyLots=BasketBaseLotsEquivalent(positionsCount);

   double distLegacy=(double)LegacyTakeProfitPts * point * (legacyLots / totalLots);
   if(distLegacy<=0.0)
      distLegacy=distChannel;

   double depthAdj=1.0 / MathMax(1.0, 1.0 + BasketTPDepthFactor * MathMax(0, positionsCount-1));
   double distChannelTight=MathMax(point * MathMax(1.0,(double)LegacyTakeProfitPts*BasketTPMinFactor), distChannel * depthAdj);

   double hybrid=MathMin(distChannelTight, distLegacy);
   hybrid=Clamp(hybrid, point, distChannel);

   return hybrid;
}

bool BasketTPHit(const string symbol,const int symIdx,const int magic,double &avgOut,double &targetOut)
{
   avgOut=0.0;
   targetOut=0.0;

   if(gPositionsCount[symIdx] <= 0) return false;
   if(gTrades[symIdx].Total() <= 0) return false;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   double avg=(gBasketAvgValid[symIdx]
               ? gBasketAvgPriceCache[symIdx]
               : BasketAvgPrice(gTrades[symIdx],gTradeLots[symIdx],mid));

   int basketDir=BasketDir(symbol,magic);
   if(basketDir==0) return false;

   double channelStep=GetCloseChannelStep(symIdx,gPositionsCount[symIdx]);
   if(channelStep<=0.0) channelStep=GetGridStepCached(symIdx);

   double dist=ComputeHybridTakeProfitDistance(symIdx,gPositionsCount[symIdx],channelStep);

   double targetPrice=(basketDir>0) ? (avg + dist) : (avg - dist);

   avgOut=avg;
   targetOut=targetPrice;

   return (basketDir>0) ? (bid >= targetPrice) : (ask <= targetPrice);
}

void CheckPairTakeProfit(string symbol,int symIdx,CArrayDouble &trades,int magic)
{
   double avg=0.0,target=0.0;
   if(BasketTPHit(symbol,symIdx,magic,avg,target))
   {
      double cp=0.0;
      int cc=0;
      ClosePositions(symbol,magic,cp,cc);
   }
}

void CheckTakeProfit()
{
   for(int i=0;i<gSymbolCount;i++)
      if(gPositionsCount[i]>0)
         CheckPairTakeProfit(gSymbols[i],i,gTrades[i],gMagics[i]);
}

void TrailingStopForPair(string symbol,int symIdx,CArrayDouble &trades,int magic)
{
   if(gPositionsCount[symIdx] <= 0) return;
   if(trades.Total()<=0) return;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double mid=0.5*(bid+ask);
   double avg=(gBasketAvgValid[symIdx]
               ? gBasketAvgPriceCache[symIdx]
               : BasketAvgPrice(trades,gTradeLots[symIdx],mid));

   int basketDir = BasketDir(symbol,magic);
   if(basketDir==0) return;

   if(basketDir>0)
   {
      if((bid-avg) <= TrailStart*point)
         return;
   }
   else
   {
      if((avg-ask) <= TrailStart*point)
         return;
   }

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      double sl=PositionGetDouble(POSITION_SL);

      if(basketDir>0)
      {
         double newSL=bid - TrailStop*point;
         if(newSL>sl || sl==0.0)
            trade.PositionModify(ticket,newSL,0.0);
      }
      else
      {
         double newSL=ask + TrailStop*point;
         if(newSL<sl || sl==0.0)
            trade.PositionModify(ticket,newSL,0.0);
      }
   }
}

void TrailingStop()
{
   if(!UseTrailingStop) return;
   for(int i=0;i<gSymbolCount;i++)
      TrailingStopForPair(gSymbols[i],i,gTrades[i],gMagics[i]);
}

bool AllowActionByPolicy(const int symIdx, const int positionsCount, const int action, const ENUM_ORDER_TYPE orderType)
{
   if(!UseDangerBrain) return true;

   BrainMode m=gMode[symIdx];

   if(DangerActionPolicy==0) return true;

   int trend = TrendDirFromH1(gSymbols[symIdx], symIdx);

   if(m==MODE_DANGER)
   {
      if(DangerActionPolicy==1)
      {
         return (action==0);
      }
      if(DangerActionPolicy==2)
      {
         if(positionsCount>0) return false;
         if(trend==0) return false;
         if(orderType==ORDER_TYPE_BUY && trend<0) return false;
         if(orderType==ORDER_TYPE_SELL && trend>0) return false;
         return true;
      }
   }

   if(m==MODE_CAUTION)
   {
      if(trend==0) return (action==0);
      if(orderType==ORDER_TYPE_BUY && trend<0) return false;
      if(orderType==ORDER_TYPE_SELL && trend>0) return false;
      return true;
   }

   return true;
}

//  Next part starts at: 

bool ComputeExactStatArbZScoreClosedBar(const string symbol,
                                        const string symbol2,
                                        const ENUM_TIMEFRAMES tf,
                                        const int period,
                                        const datetime barTime,
                                        double &zScore,
                                        int &colorCode)
{
   zScore = 0.0;
   colorCode = 0;
   if(period <= 1 || StringLen(symbol2) <= 0 || barTime <= 0)
      return false;

   double spreads[];
   ArrayResize(spreads, period);

   for(int j=0; j<period; ++j)
   {
      datetime tj = iTime(symbol, tf, j + 1);
      if(tj <= 0) return false;

      double close1 = iClose(symbol, tf, j + 1);
      double close2[];
      if(CopyClose(symbol2, tf, tj, 1, close2) <= 0)
         return false;

      double price2 = close2[0];
      if(close1 <= 0.0 || price2 <= 0.0)
         spreads[j] = 0.0;
      else
         spreads[j] = MathLog(close1) - MathLog(price2);

      if(tj == barTime)
      {
         // nothing; current closed bar should be j==0 in normal use
      }
   }

   double sum = 0.0;
   for(int j=0; j<period; ++j) sum += spreads[j];
   double mean = sum / (double)period;

   double variance_sum = 0.0;
   for(int j=0; j<period; ++j)
      variance_sum += MathPow(spreads[j] - mean, 2.0);

   double std_dev = MathSqrt(variance_sum / (double)period);
   if(std_dev > 0.0000001)
      zScore = (spreads[0] - mean) / std_dev;
   else
      zScore = 0.0;

   if(zScore >= ZScoreExtremeThreshold) colorCode = 2;
   else if(zScore <= -ZScoreExtremeThreshold) colorCode = 1;
   else colorCode = 0;

   return true;
}


int ZScoreEmergencyHedgeMagic(const int baseMagic)
{
   return baseMagic + ZScoreEmergencyHedgeMagicOffset;
}

int CountPositionsByMagicSimple(const string symbol,const int magic)
{
   int count=0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0 || !PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      count++;
   }
   return count;
}

double SumPositionLotsByMagic(const string symbol,const int magic)
{
   double lots=0.0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0 || !PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      lots += PositionGetDouble(POSITION_VOLUME);
   }
   return lots;
}

double SumPositionProfitByMagic(const string symbol,const int magic)
{
   double pnl=0.0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0 || !PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      pnl += PositionGetDouble(POSITION_PROFIT);
   }
   return pnl;
}

bool HasOpenPositionsByMagic(const string symbol,const int magic)
{
   return (CountPositionsByMagicSimple(symbol,magic) > 0);
}

bool OpenEmergencyHedgeAgainstBasket(const string symbol,const int mainMagic,const int hedgeMagic,const int basketDir)
{
   if(basketDir==0) return false;
   double totalLots = SumPositionLotsByMagic(symbol, mainMagic);
   if(totalLots <= 0.0) return false;

   double hedgeLots = NormalizeVolume(symbol, totalLots * MathMax(0.0, ZScoreEmergencyHedgeLotFactor));
   if(hedgeLots <= 0.0) return false;

   ENUM_ORDER_TYPE hedgeType = (basketDir > 0 ? ORDER_TYPE_SELL : ORDER_TYPE_BUY);
   return OpenPosition(symbol, hedgeType, hedgeLots, 0, 0, hedgeMagic);
}

void UpdateZScoreRiskGuardForSymbol(const string symbol, const int symIdx)
{
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS)
      return;

   if(!UseZScoreRiskGuard)
   {
      gZScoreLastValue[symIdx] = 0.0;
      gZScoreLastAbs[symIdx] = 0.0;
      gZScoreExtremeNow[symIdx] = false;
      gZScorePauseTrading[symIdx] = false;
      gZScoreQuietBars[symIdx] = 0;
      gZScoreLastValueA[symIdx] = 0.0;
      gZScoreLastAbsA[symIdx] = 0.0;
      gZScoreExtremeA[symIdx] = false;
      gZScoreLastValueB[symIdx] = 0.0;
      gZScoreLastAbsB[symIdx] = 0.0;
      gZScoreExtremeB[symIdx] = false;
      gZScoreEmergencyHedgeActive[symIdx] = false;
      gZScoreEmergencyOriginalDir[symIdx] = 0;
      gZScoreEmergencyHedgeBar[symIdx] = 0;
      return;
   }

   if(StringLen(ZScoreSymbol2) <= 0)
      return;

   datetime closedBarTime = iTime(symbol, ZScoreTF, 1);
   if(closedBarTime <= 0 || closedBarTime == gZScoreLastClosedBarTF[symIdx])
      return;

   gZScoreLastClosedBarTF[symIdx] = closedBarTime;

   double zA = 0.0;
   int colorA = 0;
   bool okA = ComputeExactStatArbZScoreClosedBar(symbol, ZScoreSymbol2, ZScoreTF, ZScorePeriod, closedBarTime, zA, colorA);
   if(!okA)
      return;

   gZScoreLastValueA[symIdx] = zA;
   gZScoreLastAbsA[symIdx] = MathAbs(zA);
   gZScoreExtremeA[symIdx] = (colorA == 1 || colorA == 2);

   double zB = 0.0;
   int colorB = 0;
   bool okB = false;
   if(UseSecondZScoreMarket && StringLen(ZScoreSymbol2_B) > 0)
   {
      okB = ComputeExactStatArbZScoreClosedBar(symbol, ZScoreSymbol2_B, ZScoreTF, ZScorePeriod_B, closedBarTime, zB, colorB);
      if(okB)
      {
         gZScoreLastValueB[symIdx] = zB;
         gZScoreLastAbsB[symIdx] = MathAbs(zB);
         gZScoreExtremeB[symIdx] = (colorB == 1 || colorB == 2);
      }
      else
      {
         gZScoreLastValueB[symIdx] = 0.0;
         gZScoreLastAbsB[symIdx] = 0.0;
         gZScoreExtremeB[symIdx] = false;
      }
   }
   else
   {
      gZScoreLastValueB[symIdx] = 0.0;
      gZScoreLastAbsB[symIdx] = 0.0;
      gZScoreExtremeB[symIdx] = false;
   }

   bool combinedExtreme = gZScoreExtremeA[symIdx];
   if(UseSecondZScoreMarket && StringLen(ZScoreSymbol2_B) > 0)
   {
      if(UseZScoreOrRule)
         combinedExtreme = (gZScoreExtremeA[symIdx] || gZScoreExtremeB[symIdx]);
      else
         combinedExtreme = (gZScoreExtremeA[symIdx] && gZScoreExtremeB[symIdx]);
   }

   // Effective z-score exposed to DDQN/state = the stronger of the active monitors by absolute value
   if(gZScoreLastAbsB[symIdx] > gZScoreLastAbsA[symIdx])
   {
      gZScoreLastValue[symIdx] = gZScoreLastValueB[symIdx];
      gZScoreLastAbs[symIdx]   = gZScoreLastAbsB[symIdx];
   }
   else
   {
      gZScoreLastValue[symIdx] = gZScoreLastValueA[symIdx];
      gZScoreLastAbs[symIdx]   = gZScoreLastAbsA[symIdx];
   }

   gZScoreExtremeNow[symIdx] = combinedExtreme;

   if(gZScoreExtremeNow[symIdx])
   {
      gZScorePauseTrading[symIdx] = true;
      gZScoreQuietBars[symIdx] = 0;
      gZScoreLastExtremeBarTime[symIdx] = closedBarTime;
   }
   else if(gZScorePauseTrading[symIdx])
   {
      gZScoreQuietBars[symIdx]++;
      if(gZScoreQuietBars[symIdx] >= MathMax(1, ZScoreResumeQuietBars))
      {
         gZScorePauseTrading[symIdx] = false;
         gZScoreQuietBars[symIdx] = MathMax(1, ZScoreResumeQuietBars);
      }
   }
}

bool IsZScoreTradingPaused(const int symIdx)
{
   if(!UseZScoreRiskGuard) return false;
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return false;
   return gZScorePauseTrading[symIdx];
}

void ManagePairWithDQN(string symbol,int symIdx,int &positionsCount,CArrayDouble &trades,int magic,datetime &firstTradeTime)
{
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double state[];
   datetime baseBarTimeCache = iTime(symbol, BaseTF, 1);
   double avgCacheRef = (positionsCount>0 && trades.Total()>0 ? BasketAvgPrice(trades,gTradeLots[symIdx],0.5*(bid+ask)) : 0.5*(bid+ask));
   int basketDirCacheNow = (positionsCount>0 ? BasketDir(symbol,magic) : 0);
   double lastEntryCacheRef = GetLastEntryPriceState(symbol, magic, basketDirCacheNow, avgCacheRef);
   long avgQCache = QuantizePriceByPoint(avgCacheRef, point);
   long lastEntryQCache = QuantizePriceByPoint(lastEntryCacheRef, point);

   bool usedStateCache=false;
   if(ShouldUseFastStateCache() && symIdx>=0 && symIdx<MAX_SYMBOLS)
   {
      if(gStateCache[symIdx].valid && gStateCache[symIdx].symbol==symbol && gStateCache[symIdx].baseBarTime==baseBarTimeCache &&
         gStateCache[symIdx].positionsCount==positionsCount && gStateCache[symIdx].basketDir==basketDirCacheNow &&
         gStateCache[symIdx].avgQ==avgQCache && gStateCache[symIdx].lastEntryQ==lastEntryQCache)
      {
         CloneState(gStateCache[symIdx].state, state);
         usedStateCache=true;
      }
   }

   if(!usedStateCache)
   {
      BuildState(symbol,symIdx,positionsCount,trades,state);
      if(ShouldUseFastStateCache() && symIdx>=0 && symIdx<MAX_SYMBOLS)
      {
         gStateCache[symIdx].valid=true;
         gStateCache[symIdx].symbol=symbol;
         gStateCache[symIdx].baseBarTime=baseBarTimeCache;
         gStateCache[symIdx].positionsCount=positionsCount;
         gStateCache[symIdx].basketDir=basketDirCacheNow;
         gStateCache[symIdx].avgQ=avgQCache;
         gStateCache[symIdx].lastEntryQ=lastEntryQCache;
         CloneState(state, gStateCache[symIdx].state);
      }
   }

   bool extreme=IsExtremeState(symIdx,positionsCount);
   if(SeparateExtremeStates)
   {
      int sz=ArraySize(state);
      if(sz>0) state[sz-1]=(extreme?1.0:0.0);
   }

   double atrRatio = GetAtrRatioCached_Base(symbol);
   int activeRegime = (UseRegimeBank ? RegimeIndexFromRatio(atrRatio) : 0);

   int wantIn = ArraySize(state);
   if(gDQN[symIdx][activeRegime].input_dim != wantIn)
   {
      for(int r=0;r<REGIME_COUNT;r++)
         InitOrRandomizeDQN(symIdx,r,wantIn);
   }

   double qNow[];
   DQNForward(symIdx,activeRegime,state,qNow);

   int basketDirNow = (positionsCount>0 ? BasketDir(symbol,magic) : 0);
   CaptureBasketSnapshot(symbol,symIdx,magic,activeRegime,basketDirNow,positionsCount,state,qNow);
   UpdateDDEventLifecycle(symIdx, activeRegime, basketDirNow, state, qNow);

   AccrueDenseBasketHealthFeedback(symbol, symIdx, magic, positionsCount, extreme);

   UpdateForcedEntryWatchdog(symIdx);
   bool forceEntryNow = ForcedEntryRequiredNow(symIdx);

   DecisionSupportContext liveSupport;
   BuildDecisionSupportContext(symIdx, activeRegime, state, qNow, forceEntryNow, basketDirNow, liveSupport);

   double combinedTrend=0.0;
   double reversalRisk=0.0;
   ComputeLiveTrendBiasAndReversal(symbol, symIdx, activeRegime, liveSupport, combinedTrend, reversalRisk);

   bool zscoreTradingPaused = (ZScoreBlockAllTrading && IsZScoreTradingPaused(symIdx));

   if(UseZScoreRiskGuard && ZScoreBlockAllTrading && UseZScoreCloseSmallBasket)
   {
      int closeMaxTrades = MathMax(0, ZScoreCloseBasketMaxTrades);
      bool smallBasket = (positionsCount > 0 && positionsCount <= closeMaxTrades);
      bool freshExtremeSignal = (gZScoreExtremeNow[symIdx] &&
                                 gZScoreLastClosedBarTF[symIdx] > 0 &&
                                 gZScoreCloseHandledBar[symIdx] != gZScoreLastClosedBarTF[symIdx]);

      if(smallBasket && freshExtremeSignal)
      {
         double closedProfitRisk = 0.0;
         int closedCountRisk = 0;
         if(ClosePositions(symbol, magic, closedProfitRisk, closedCountRisk))
         {
            gZScoreCloseHandledBar[symIdx] = gZScoreLastClosedBarTF[symIdx];
            CountOpenPositions();
            positionsCount = gPositionsCount[symIdx];
            firstTradeTime = gFirstTradeTime[symIdx];
            gDecisionSupportCache[symIdx].valid = false;

            if(positionsCount == 0)
            {
               firstTradeTime = 0;
               ResetDenseBasketHealthTracker(symIdx);
            }
            return;
         }

      }
   }

   int hedgeMagic = ZScoreEmergencyHedgeMagic(magic);
   bool emergencyHedgeActive = HasOpenPositionsByMagic(symbol, hedgeMagic);
   if(!emergencyHedgeActive && gZScoreEmergencyHedgeActive[symIdx])
   {
      gZScoreEmergencyHedgeActive[symIdx] = false;
      gZScoreEmergencyOriginalDir[symIdx] = 0;
      gZScoreEmergencyHedgeBar[symIdx] = 0;
   }
   else if(emergencyHedgeActive)
   {
      gZScoreEmergencyHedgeActive[symIdx] = true;
      if(gZScoreEmergencyOriginalDir[symIdx]==0 && basketDirNow!=0)
         gZScoreEmergencyOriginalDir[symIdx] = basketDirNow;
   }

   bool freshExtremeSignal = (gZScoreExtremeNow[symIdx] &&
                              gZScoreLastClosedBarTF[symIdx] > 0 &&
                              gZScoreCloseHandledBar[symIdx] != gZScoreLastClosedBarTF[symIdx]);

   if(UseZScoreRiskGuard && ZScoreBlockAllTrading && UseZScoreEmergencyHedge && freshExtremeSignal)
   {
      int closeMaxTrades = MathMax(0, ZScoreCloseBasketMaxTrades);
      bool deepBasket = (positionsCount > closeMaxTrades);
      if(deepBasket && positionsCount > 0 && basketDirNow != 0 && !emergencyHedgeActive)
      {
         if(OpenEmergencyHedgeAgainstBasket(symbol, magic, hedgeMagic, basketDirNow))
         {
            gZScoreEmergencyHedgeActive[symIdx] = true;
            gZScoreEmergencyOriginalDir[symIdx] = basketDirNow;
            gZScoreEmergencyHedgeBar[symIdx] = gZScoreLastClosedBarTF[symIdx];
            gZScoreCloseHandledBar[symIdx] = gZScoreLastClosedBarTF[symIdx];
            emergencyHedgeActive = true;
         }
      }
   }

   if(gZScoreEmergencyHedgeActive[symIdx])
   {
      emergencyHedgeActive = HasOpenPositionsByMagic(symbol, hedgeMagic);
      if(!emergencyHedgeActive)
      {
         gZScoreEmergencyHedgeActive[symIdx] = false;
         gZScoreEmergencyOriginalDir[symIdx] = 0;
         gZScoreEmergencyHedgeBar[symIdx] = 0;
      }
   }

   double hedgePnL = (gZScoreEmergencyHedgeActive[symIdx] ? SumPositionProfitByMagic(symbol, hedgeMagic) : 0.0);
   double mainPnL = SumPositionProfitByMagic(symbol, magic);
   double combinedPnL = mainPnL + hedgePnL;
   double combinedCloseTarget = (RecoveryCloseAtBreakevenOnly ? 0.0 : RecoveryCombinedCloseMoney);

   if(gZScoreEmergencyHedgeActive[symIdx] && combinedPnL >= combinedCloseTarget)
   {
      double c1=0.0,c2=0.0;
      int n1=0,n2=0;
      bool ok1=ClosePositions(symbol, magic, c1, n1);
      bool ok2=ClosePositions(symbol, hedgeMagic, c2, n2);
      if(ok1 || ok2)
      {
         gZScoreEmergencyHedgeActive[symIdx] = false;
         gZScoreEmergencyOriginalDir[symIdx] = 0;
         gZScoreEmergencyHedgeBar[symIdx] = 0;
         CountOpenPositions();
         positionsCount = gPositionsCount[symIdx];
         firstTradeTime = gFirstTradeTime[symIdx];
         gDecisionSupportCache[symIdx].valid = false;
         if(positionsCount == 0)
         {
            firstTradeTime = 0;
            ResetDenseBasketHealthTracker(symIdx);
         }
         return;
      }
   }

   bool recoveryAddsAllowed = (!gZScoreEmergencyHedgeActive[symIdx] ||
                               (UseRecoveryAfterEmergencyHedge &&
                                !zscoreTradingPaused &&
                                gZScoreQuietBars[symIdx] >= MathMax(1, RecoveryRestartQuietBars)));
   bool blockNewFirstEntries = (zscoreTradingPaused || gZScoreEmergencyHedgeActive[symIdx]);

   //==========================================================
   // 1) NO OPEN BASKET -> DQN decides whether to open first leg
   //==========================================================
   if(positionsCount==0)
   {
      if(blockNewFirstEntries)
         return;

      if(IsReentryCooldownActive(symbol, symIdx) &&
         !(forceEntryNow && ForceEntryBypassesReentryCooldown))
      {
         return;
      }

      int action = (forceEntryNow
                    ? DQNSelectForcedEntryAction(symIdx,activeRegime,state)
                    : DQNSelectAction(symIdx,activeRegime,state));

      ENUM_ORDER_TYPE orderType=ORDER_TYPE_BUY;
      bool shouldTrade=false;

      if(action==1){ orderType=ORDER_TYPE_BUY;  shouldTrade=true; }
      else if(action==2){ orderType=ORDER_TYPE_SELL; shouldTrade=true; }

      if(zscoreTradingPaused)
         shouldTrade=false;

      // Normal mode respects danger policy.
      // Forced-entry mode can bypass first-entry blocking if configured.
      if(!forceEntryNow || !ForcedEntryIgnoreDanger)
      {
         if(UseDangerBrain && DangerActionPolicy==0)
         {
            BrainMode m=gMode[symIdx];
            if(m==MODE_DANGER && DangerBlocksNewEntries)
               shouldTrade=false;
         }

         if(UseDangerBrain && DangerActionPolicy!=0)
         {
            if(!AllowActionByPolicy(symIdx, positionsCount, action, orderType))
               shouldTrade=false;
         }
      }

      if(shouldTrade)
      {
         if(!PassesOpenIntervalGate(symbol, symIdx))
         {
            if(VerboseLogging)
               Print("ENTRY BLOCKED | open interval gate | symbol=", symbol,
                     " secSinceLast=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]));
            shouldTrade=false;
         }
      }

      if(shouldTrade)
      {
         double baseVol0=NormalizeVolume(symbol, Lots*GetEquityBudgetScale());
         double vol0=baseVol0;
         bool deepBasketStrongFirstTradeArmed=ShouldUseDeepBasketStrongFirstTrade(symIdx);

         if(deepBasketStrongFirstTradeArmed)
         {
            vol0=NormalizeVolume(symbol, baseVol0*DeepBasketStrongFirstTradeMult);

            if(VerboseLogging)
               Print("DEEP BASKET STRONG FIRST TRADE USED | symbol=", symbol,
                     " closedCount=", gDeepBasketStrongEntryClosedCount[symIdx],
                     " baseVol=", DoubleToString(baseVol0,2),
                     " boostedVol=", DoubleToString(vol0,2),
                     " mult=", DoubleToString(DeepBasketStrongFirstTradeMult,2));
         }

         if(!blockNewFirstEntries && OpenPosition(symbol,orderType,vol0,0,0,magic))
         {
            if(deepBasketStrongFirstTradeArmed)
               ClearDeepBasketStrongFirstTradeArm(symIdx);

            int positionsAfter=0;
            int basketDirAfter=0;
            double entryPrice0=0.0;
            double entryVol0=vol0;
            double basketAvg0=0.0;
            datetime entryTime0=TimeCurrent();

            if(BuildPostTradeSnapshot(symbol,
                                      symIdx,
                                      magic,
                                      orderType,
                                      vol0,
                                      positionsAfter,
                                      basketDirAfter,
                                      entryPrice0,
                                      entryVol0,
                                      basketAvg0,
                                      entryTime0))
            {
               positionsCount = positionsAfter;
               firstTradeTime = gFirstTradeTime[symIdx];
               if(firstTradeTime==0) firstTradeTime=entryTime0;

               MarkTradeOpened(symIdx, entryTime0);

               gBadHaveBasketFP[symIdx]=false;
               gBadHaveMiniBasket[symIdx]=false;

               PendingAdd(symIdx,
                          activeRegime,
                          action,
                          basketDirAfter,
                          positionsAfter,
                          state,
                          entryPrice0,
                          entryVol0,
                          positionsAfter,
                          basketAvg0,
                          entryTime0);

               if(isTraining && AllowLearnEntryOrAveraging(symIdx))
               {
                  double reward=ComputeOpenRewardV2(symIdx,
                                                    (orderType==ORDER_TYPE_BUY),
                                                    0,
                                                    combinedTrend,
                                                    reversalRisk,
                                                    extreme);
                  double nextStateAfterOpen[];
                  BuildPostOpenNextState(symbol, symIdx, positionsAfter, gTrades[symIdx], nextStateAfterOpen);
                  SubmitTransitionWithNextState(symIdx,activeRegime,state,action,reward,false,nextStateAfterOpen);
               }
            }
            else
            {
               positionsCount = gPositionsCount[symIdx];
               firstTradeTime = gFirstTradeTime[symIdx];
               if(firstTradeTime==0) firstTradeTime=TimeCurrent();
            }
         }

         Print("NO BASKET | action=", action,
               " epsilon=", DoubleToString(currentEpsilon,4),
               " danger=", DoubleToString(gPDanger[symIdx],4),
               " mode=", (int)gMode[symIdx],
               " forceEntry=", (forceEntryNow ? "YES" : "NO"),
               " lastOpenAgoSec=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]));
      }
   }
   //==========================================================
   // 2) OPEN BASKET -> ORIGINAL GRID ADD LOGIC ONLY
   //==========================================================
   else
   {
      if(positionsCount < MaxTrades)
      {
         int basketDirLive = BasketDir(symbol,magic);
         if(basketDirLive!=0)
         {
            ENUM_POSITION_TYPE pt = (basketDirLive>0 ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);

            double buyLevel=0.0, sellLevel=0.0;
            CalculateLevelsSimple(symbol,symIdx,positionsCount,trades,magic,buyLevel,sellLevel);

            if(pt==POSITION_TYPE_BUY)
            {
               double candidateBuyPrice=SymbolInfoDouble(symbol,SYMBOL_ASK);
               if(candidateBuyPrice<=0.0) candidateBuyPrice=ask;

               if(bid<=buyLevel)
               {
                  if(!PassesOpenIntervalGate(symbol, symIdx))
                  {
                     if(VerboseLogging)
                        Print("ADD BLOCKED | open interval gate | symbol=", symbol,
                              " secSinceLast=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]),
                              " candidate=", DoubleToString(candidateBuyPrice,_Digits));
                  }
                  else
                  {
                     double requiredAddGap=0.0, actualAddGap=0.0, lastAddPrice=0.0;
                     if(!PassesMinAddSpacingGate(symbol, symIdx, magic, basketDirLive, candidateBuyPrice,
                                                 requiredAddGap, actualAddGap, lastAddPrice))
                     {
                        if(VerboseLogging)
                           Print("ADD BLOCKED | spacing gate | symbol=", symbol,
                                 " candidate=", DoubleToString(candidateBuyPrice,_Digits),
                                 " buyLevel=", DoubleToString(buyLevel,_Digits),
                                 " lastEntry=", DoubleToString(lastAddPrice,_Digits),
                                 " actualGap=", DoubleToString(actualAddGap,_Digits),
                                 " requiredGap=", DoubleToString(requiredAddGap,_Digits));
                     }
                     else
                     {
                        int preOpenCount = positionsCount;
                        double smartSpacingMult=1.0;
                        double smartLotScale=1.0;
                        double smartAddRisk=0.0;
                        bool smartAddOk=EvaluateSmartAddGate(symbol,
                                                             symIdx,
                                                             activeRegime,
                                                             state,
                                                             basketDirLive,
                                                             preOpenCount,
                                                             combinedTrend,
                                                             reversalRisk,
                                                             extreme,
                                                             smartSpacingMult,
                                                             smartLotScale,
                                                             smartAddRisk);

                        double basketAvgLive=BasketAvgPrice(trades,gTradeLots[symIdx],candidateBuyPrice);
                        double baseStep=MathMax(0.0,basketAvgLive-buyLevel);
                        double gatedBuyLevel=(smartSpacingMult>1.0 ? basketAvgLive-baseStep*smartSpacingMult : buyLevel);

                        if(!smartAddOk)
                        {
                           if(VerboseLogging)
                              Print("ADD BLOCKED | smart gate | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " buyLevel=", DoubleToString(buyLevel,_Digits));
                        }
                        else if(bid>gatedBuyLevel)
                        {
                           if(VerboseLogging)
                              Print("ADD DELAYED | widened buy level | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " bid=", DoubleToString(bid,_Digits),
                                    " gatedBuyLevel=", DoubleToString(gatedBuyLevel,_Digits));
                        }
                        else
                        {
                           double vol=NormalizeVolume(symbol,CalculateLot(symbol,preOpenCount)*smartLotScale);

                           if(recoveryAddsAllowed && OpenPosition(symbol,ORDER_TYPE_BUY,vol,0,0,magic))
                           {
                              int positionsAfter=0;
                              int basketDirAfter=0;
                              double entryPriceBuy=0.0;
                              double entryVolBuy=vol;
                              double basketAvgBuy=0.0;
                              datetime entryTimeBuy=TimeCurrent();

                              if(BuildPostTradeSnapshot(symbol,
                                                        symIdx,
                                                        magic,
                                                        ORDER_TYPE_BUY,
                                                        vol,
                                                        positionsAfter,
                                                        basketDirAfter,
                                                        entryPriceBuy,
                                                        entryVolBuy,
                                                        basketAvgBuy,
                                                        entryTimeBuy))
                              {
                                 positionsCount = positionsAfter;
                                 if(firstTradeTime==0)
                                    firstTradeTime = gFirstTradeTime[symIdx];

                                 MarkTradeOpened(symIdx, entryTimeBuy);

                                 int learnAction = 1;

                                 PendingAdd(symIdx,
                                            activeRegime,
                                            learnAction,
                                            basketDirAfter,
                                            positionsAfter,
                                            state,
                                            entryPriceBuy,
                                            entryVolBuy,
                                            positionsAfter,
                                            basketAvgBuy,
                                            entryTimeBuy);

                                 if(isTraining && AllowLearnEntryOrAveraging(symIdx))
                                 {
                                    double reward=ComputeOpenRewardV2(symIdx,true,preOpenCount,combinedTrend,reversalRisk,extreme);
                                    double nextStateAfterOpen[];
                                    BuildPostOpenNextState(symbol, symIdx, positionsAfter, gTrades[symIdx], nextStateAfterOpen);
                                    SubmitTransitionWithNextState(symIdx,activeRegime,state,learnAction,reward,false,nextStateAfterOpen);
                                 }
                              }
                              else
                              {
                                 positionsCount = gPositionsCount[symIdx];
                              }
                           }
                        }
                     }
                  }
               }
            }
            else
            {
               double candidateSellPrice=SymbolInfoDouble(symbol,SYMBOL_BID);
               if(candidateSellPrice<=0.0) candidateSellPrice=bid;

               if(ask>=sellLevel)
               {
                  if(!PassesOpenIntervalGate(symbol, symIdx))
                  {
                     if(VerboseLogging)
                        Print("ADD BLOCKED | open interval gate | symbol=", symbol,
                              " secSinceLast=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]),
                              " candidate=", DoubleToString(candidateSellPrice,_Digits));
                  }
                  else
                  {
                     double requiredAddGap=0.0, actualAddGap=0.0, lastAddPrice=0.0;
                     if(!PassesMinAddSpacingGate(symbol, symIdx, magic, basketDirLive, candidateSellPrice,
                                                 requiredAddGap, actualAddGap, lastAddPrice))
                     {
                        if(VerboseLogging)
                           Print("ADD BLOCKED | spacing gate | symbol=", symbol,
                                 " candidate=", DoubleToString(candidateSellPrice,_Digits),
                                 " sellLevel=", DoubleToString(sellLevel,_Digits),
                                 " lastEntry=", DoubleToString(lastAddPrice,_Digits),
                                 " actualGap=", DoubleToString(actualAddGap,_Digits),
                                 " requiredGap=", DoubleToString(requiredAddGap,_Digits));
                     }
                     else
                     {
                        int preOpenCount = positionsCount;
                        double smartSpacingMult=1.0;
                        double smartLotScale=1.0;
                        double smartAddRisk=0.0;
                        bool smartAddOk=EvaluateSmartAddGate(symbol,
                                                             symIdx,
                                                             activeRegime,
                                                             state,
                                                             basketDirLive,
                                                             preOpenCount,
                                                             combinedTrend,
                                                             reversalRisk,
                                                             extreme,
                                                             smartSpacingMult,
                                                             smartLotScale,
                                                             smartAddRisk);

                        double basketAvgLive=BasketAvgPrice(trades,gTradeLots[symIdx],candidateSellPrice);
                        double baseStep=MathMax(0.0,sellLevel-basketAvgLive);
                        double gatedSellLevel=(smartSpacingMult>1.0 ? basketAvgLive+baseStep*smartSpacingMult : sellLevel);

                        if(!smartAddOk)
                        {
                           if(VerboseLogging)
                              Print("ADD BLOCKED | smart gate | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " sellLevel=", DoubleToString(sellLevel,_Digits));
                        }
                        else if(ask<gatedSellLevel)
                        {
                           if(VerboseLogging)
                              Print("ADD DELAYED | widened sell level | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " ask=", DoubleToString(ask,_Digits),
                                    " gatedSellLevel=", DoubleToString(gatedSellLevel,_Digits));
                        }
                        else
                        {
                           double vol=NormalizeVolume(symbol,CalculateLot(symbol,preOpenCount)*smartLotScale);

                           if(recoveryAddsAllowed && OpenPosition(symbol,ORDER_TYPE_SELL,vol,0,0,magic))
                           {
                              int positionsAfter=0;
                              int basketDirAfter=0;
                              double entryPriceSell=0.0;
                              double entryVolSell=vol;
                              double basketAvgSell=0.0;
                              datetime entryTimeSell=TimeCurrent();

                              if(BuildPostTradeSnapshot(symbol,
                                                        symIdx,
                                                        magic,
                                                        ORDER_TYPE_SELL,
                                                        vol,
                                                        positionsAfter,
                                                        basketDirAfter,
                                                        entryPriceSell,
                                                        entryVolSell,
                                                        basketAvgSell,
                                                        entryTimeSell))
                              {
                                 positionsCount = positionsAfter;
                                 if(firstTradeTime==0)
                                    firstTradeTime = gFirstTradeTime[symIdx];

                                 MarkTradeOpened(symIdx, entryTimeSell);

                                 int learnAction = 2;

                                 PendingAdd(symIdx,
                                            activeRegime,
                                            learnAction,
                                            basketDirAfter,
                                            positionsAfter,
                                            state,
                                            entryPriceSell,
                                            entryVolSell,
                                            positionsAfter,
                                            basketAvgSell,
                                            entryTimeSell);

                                 if(isTraining && AllowLearnEntryOrAveraging(symIdx))
                                 {
                                    double reward=ComputeOpenRewardV2(symIdx,false,preOpenCount,combinedTrend,reversalRisk,extreme);
                                    double nextStateAfterOpen[];
                                    BuildPostOpenNextState(symbol, symIdx, positionsAfter, gTrades[symIdx], nextStateAfterOpen);
                                    SubmitTransitionWithNextState(symIdx,activeRegime,state,learnAction,reward,false,nextStateAfterOpen);
                                 }
                              }
                              else
                              {
                                 positionsCount = gPositionsCount[symIdx];
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //==========================================================
   // 3) HYBRID BASKET TP
   //==========================================================
   if(positionsCount>0 && !gZScoreEmergencyHedgeActive[symIdx])
   {
      int basketDirLive=BasketDir(symbol,magic);
      if(basketDirLive!=0)
      {
         double avgTP=0.0;
         double targetPrice=0.0;
         bool shouldClose=BasketTPHit(symbol,symIdx,magic,avgTP,targetPrice);

         if(shouldClose)
         {
            int positionsBeforeClose=positionsCount;

            BasketCloseResult closeRes;
            PositionCloseItem closedItems[];
            if(ClosePositionsDetailed(symbol,magic,closeRes,closedItems))
            {
               CountOpenPositions();
               positionsCount = gPositionsCount[symIdx];
               firstTradeTime = gFirstTradeTime[symIdx];
               gDecisionSupportCache[symIdx].valid=false;

               if(closeRes.allClosed)
               {
                  firstTradeTime=0;
                  ResetDenseBasketHealthTracker(symIdx);
               }

               if(isTraining && AllowLearnClose(symIdx))
               {
                  double reward = ComputeCloseRewardV2(symIdx, closeRes, positionsBeforeClose, extreme);

                  double nextState[];
                  BuildCloseNextState(symbol,symIdx,positionsCount,nextState);

                  double resolvedRewardSum=0.0;
                  int resolvedTransitions=PendingResolveForSymbolClose(symIdx, reward, closeRes.allClosed, nextState, closedItems, resolvedRewardSum);
                  if(resolvedTransitions>0)
                  {
                     episodeCount += resolvedTransitions;
                     totalReward += resolvedRewardSum;
                  }
               }
            }
         }
      }
   }

   if(ShouldTrainReplayBatchNow(symIdx, symbol))
      TrainReplayBatch();
}

void DetectZones(string symbol)
{
   int bars = iBars(symbol, BaseTF);
   if(bars <= 0) return;

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   int minBase = MathMax(1, MinZoneBaseBars);
   int maxBase = MathMax(minBase, MaxZoneBaseBars);
   if(bars < maxBase + 2) return;

   int breakoutShift = 1;
   datetime lastClosedTime = iTime(symbol, BaseTF, breakoutShift);

   int added=0;
   int maxAdd=MathMax(1, MaxZonesPerBar);

   for(int baseLen=minBase; baseLen<=maxBase && added<maxAdd; baseLen++)
   {
      int startIndex = breakoutShift + baseLen;
      if(startIndex>=bars) continue;

      double highPrice=iHigh(symbol,BaseTF,startIndex);
      double lowPrice =iLow (symbol,BaseTF,startIndex);

      bool consolidated=true;
      for(int iBar=startIndex-1; iBar>breakoutShift; iBar--)
      {
         double h=iHigh(symbol,BaseTF,iBar);
         double l=iLow (symbol,BaseTF,iBar);
         if(h>highPrice) highPrice=h;
         if(l<lowPrice)  lowPrice=l;

         if((highPrice-lowPrice) > ZoneMaxHeightPoints*point)
         { consolidated=false; break; }
      }
      if(!consolidated) continue;

      double closePrice=iClose(symbol,BaseTF,breakoutShift);
      double breakoutLow=iLow(symbol,BaseTF,breakoutShift);
      double breakoutHigh=iHigh(symbol,BaseTF,breakoutShift);

      bool isDemand=(closePrice>highPrice && breakoutLow>=lowPrice);
      bool isSupply=(closePrice<lowPrice && breakoutHigh<=highPrice);
      if(!isDemand && !isSupply) continue;

      bool overlaps=false, duplicate=false;
      int totalZones=ArraySize(zones);
      for(int j=0;j<totalZones;j++)
      {
         if(zones[j].symbol!=symbol) continue;
         if(lastClosedTime < zones[j].endTime)
         {
            double maxLow=MathMax(lowPrice, zones[j].low);
            double minHigh=MathMin(highPrice, zones[j].high);
            if(maxLow<=minHigh){ overlaps=true; break; }
            if(MathAbs(zones[j].high-highPrice)<point &&
               MathAbs(zones[j].low -lowPrice )<point)
            { duplicate=true; break; }
         }
      }
      if(overlaps||duplicate) continue;

      int zc=ArraySize(zones);
      if(zc>=MaxZonesTracked && zc>0)
      {
         if(DrawZonesOnChart)
         {
            ObjectDelete(0,zones[0].name);
            ObjectDelete(0,zones[0].name+"_lbl");
         }
         ArrayRemove(zones,0,1);
         zc--;
      }

      ArrayResize(zones, zc+1);
      zones[zc].symbol=symbol;
      zones[zc].high=highPrice;
      zones[zc].low=lowPrice;
      zones[zc].startTime=iTime(symbol,BaseTF,(startIndex>0?startIndex:breakoutShift+minBase));
      zones[zc].endTime=TimeCurrent() + (datetime)PeriodSeconds(BaseTF)*ZoneExtendBars;
      zones[zc].breakoutTime=lastClosedTime;
      zones[zc].isDemand=isDemand;
      zones[zc].tested=false;
      zones[zc].broken=false;
      zones[zc].name="SDZone_"+symbol+"_"+IntegerToString(zc)+"_"+TimeToString(zones[zc].startTime,TIME_DATE|TIME_SECONDS);

      added++;
   }
}

void UpdateZones(string symbol)
{
   int total=ArraySize(zones);
   if(total<=0) return;

   double prevHigh=iHigh(symbol,BaseTF,1);
   double prevLow =iLow (symbol,BaseTF,1);
   double prevClose=iClose(symbol,BaseTF,1);
   datetime lastClosedTime=iTime(symbol,BaseTF,1);

   for(int i=total-1;i>=0;i--)
   {
      if(zones[i].symbol!=symbol) continue;

      if(lastClosedTime>=zones[i].endTime)
      {
         if(DrawZonesOnChart)
         {
            ObjectDelete(0,zones[i].name);
            ObjectDelete(0,zones[i].name+"_lbl");
         }
         ArrayRemove(zones,i,1);
         continue;
      }

      bool overlap=(prevLow<=zones[i].high && prevHigh>=zones[i].low);
      if(overlap) zones[i].tested=true;

      if(zones[i].isDemand)
      {
         if(prevClose<zones[i].low) zones[i].broken=true;
      }
      else
      {
         if(prevClose>zones[i].high) zones[i].broken=true;
      }

      if(DrawZonesOnChart)
      {
         color zoneColor;
         if(zones[i].broken) zoneColor=clrDarkGray;
         else if(zones[i].tested) zoneColor=(zones[i].isDemand?clrBlueViolet:clrOrange);
         else zoneColor=(zones[i].isDemand?clrBlue:clrRed);

         ObjectDelete(0,zones[i].name);
         ObjectCreate(0,zones[i].name,OBJ_RECTANGLE,0,zones[i].startTime,zones[i].high,zones[i].endTime,zones[i].low);
         ObjectSetInteger(0,zones[i].name,OBJPROP_COLOR,zoneColor);
         ObjectSetInteger(0,zones[i].name,OBJPROP_FILL,true);
         ObjectSetInteger(0,zones[i].name,OBJPROP_BACK,true);

         datetime midTime=zones[i].startTime+(zones[i].endTime-zones[i].startTime)/2;
         double midPrice=(zones[i].high+zones[i].low)/2.0;

         string txt=(zones[i].isDemand?"Demand":"Supply");
         if(zones[i].tested) txt+=" (T)";
         if(zones[i].broken) txt+=" (X)";

         string labelName=zones[i].name+"_lbl";
         ObjectDelete(0,labelName);
         ObjectCreate(0,labelName,OBJ_TEXT,0,midTime,midPrice);
         ObjectSetString(0,labelName,OBJPROP_TEXT,txt);
         ObjectSetInteger(0,labelName,OBJPROP_COLOR,clrBlack);
         ObjectSetInteger(0,labelName,OBJPROP_ANCHOR,ANCHOR_CENTER);
      }
   }
}

string TFLabel(const ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      default:         return IntegerToString((int)tf);
   }
}

string StructVizPrefix(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return "DDQN_STRUCT_" + symbol + "_" + TFLabel(tf) + "_";
}

void ClearStructureVisualizationForTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   string pfx = StructVizPrefix(symbol, tf);
   ObjectsDeleteAll(0, pfx);
}

void DrawSwingPointLabel(const string name, const datetime when, const double price, const string txt, const color clr)
{
   ObjectDelete(0, name);
   if(!ObjectCreate(0, name, OBJ_TEXT, 0, when, price))
      return;
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
}

void DrawZoneRectLabel(const string rectName, const string txtName, const datetime t1, const datetime t2, const double high, const double low, const color clr, const string txt)
{
   ObjectDelete(0, rectName);
   if(ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, t1, high, t2, low))
   {
      ObjectSetInteger(0, rectName, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, rectName, OBJPROP_FILL, true);
      ObjectSetInteger(0, rectName, OBJPROP_BACK, true);
      ObjectSetInteger(0, rectName, OBJPROP_WIDTH, 1);
   }

   ObjectDelete(0, txtName);
   datetime tm = t1 + (t2 - t1) / 2;
   double mp = 0.5 * (high + low);
   if(ObjectCreate(0, txtName, OBJ_TEXT, 0, tm, mp))
   {
      ObjectSetString(0, txtName, OBJPROP_TEXT, txt);
      ObjectSetInteger(0, txtName, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, txtName, OBJPROP_ANCHOR, ANCHOR_CENTER);
      ObjectSetInteger(0, txtName, OBJPROP_FONTSIZE, 8);
   }
}

void DrawCandleFlagLabel(const string name, const datetime when, const double price, const string txt, const color clr)
{
   ObjectDelete(0, name);
   if(!ObjectCreate(0, name, OBJ_TEXT, 0, when, price))
      return;
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
}

void UpdateStructureVisualizationForTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   if(!(DrawSwingStructureOnChart || DrawSwingZonesOnChart || DrawZoneCandleFlagsOnChart))
      return;
   if(symbol != _Symbol)
      return;

   string pfx = StructVizPrefix(symbol, tf);
   ClearStructureVisualizationForTF(symbol, tf);

   if(DrawSwingStructureOnChart)
   {
      SwingPointMem swings[];
      int swingCount = 0;
      if(BuildSwingMemoryForTF_Core(symbol, tf, swings, swingCount))
      {
         int drawCount = MathMin(swingCount, MathMax(MaxDrawnSwingsPerTF, 1));
         int start = MathMax(0, swingCount - drawCount);
         int n = 0;
         for(int i=start; i<swingCount; ++i)
         {
            string nm = pfx + "SW_" + IntegerToString(n);
            string lbl = (swings[i].isHigh ? "SH" : "SL");
            color clr = (swings[i].isHigh ? clrTomato : clrLimeGreen);
            DrawSwingPointLabel(nm, swings[i].when, swings[i].price, TFLabel(tf) + " " + lbl, clr);
            n++;
         }
      }
   }

   SwingDerivedZone demands[], supplies[];
   int demandCount = 0, supplyCount = 0;
   bool haveZones = BuildSwingDerivedZonesForTF(symbol, tf, demands, demandCount, supplies, supplyCount);

   if(DrawSwingZonesOnChart && haveZones)
   {
      double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
      int idxDem = -1, idxSup = -1;
      double ref = GetCloseSafe(symbol, tf, 1);
      SelectBestSwingZone(demands, demandCount, symbol, tf, ref, ref, ref, atr, idxDem);
      SelectBestSwingZone(supplies, supplyCount, symbol, tf, ref, ref, ref, atr, idxSup);

      int drawn = 0;
      if(idxDem >= 0 && drawn < MaxDrawnZonesPerTF)
      {
         SwingDerivedZone z = demands[idxDem];
         datetime t1 = z.pivotTime;
         datetime t2 = TimeCurrent() + (datetime)PeriodSeconds(tf) * 40;
         string tag = (z.lifeState > 0 ? "Active" : (z.lifeState == 0 ? "Candidate" : "Invalid"));
         DrawZoneRectLabel(pfx + "ZD_RECT_0", pfx + "ZD_TXT_0", t1, t2, z.high, z.low, clrLimeGreen, TFLabel(tf) + " Demand " + tag);
         drawn++;
      }
      if(idxSup >= 0 && drawn < MaxDrawnZonesPerTF)
      {
         SwingDerivedZone z = supplies[idxSup];
         datetime t1 = z.pivotTime;
         datetime t2 = TimeCurrent() + (datetime)PeriodSeconds(tf) * 40;
         string tag = (z.lifeState > 0 ? "Active" : (z.lifeState == 0 ? "Candidate" : "Invalid"));
         DrawZoneRectLabel(pfx + "ZS_RECT_0", pfx + "ZS_TXT_0", t1, t2, z.high, z.low, clrTomato, TFLabel(tf) + " Supply " + tag);
      }
   }

   if(DrawZoneCandleFlagsOnChart)
   {
      ZoneBranchFeaturePack zf;
      ComputeZoneBranchFeatures(symbol, tf, GetCloseSafe(symbol, tf, 1), GetCloseSafe(symbol, tf, 1), GetCloseSafe(symbol, tf, 1), zf);

      double open1 = iOpen(symbol, tf, 1);
      double high1 = iHigh(symbol, tf, 1);
      double low1  = iLow(symbol, tf, 1);
      double close1= GetCloseSafe(symbol, tf, 1);
      double range = MathMax(high1 - low1, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);
      double body  = MathAbs(close1 - open1);
      double upper = high1 - MathMax(open1, close1);
      double lower = MathMin(open1, close1) - low1;
      double closeLoc = Clamp((close1 - low1) / range, 0.0, 1.0);
      double activeDist = (zf.zoneSideState > 0.0 ? MathAbs(zf.nearestDemandDist) : MathAbs(zf.nearestSupplyDist));
      double zoneNearness = 1.0 - Clamp(activeDist, 0.0, 1.0);
      double zoneDepthCentrality = 1.0 - MathAbs(zf.zoneDepthPosition);
      double zoneContext = Clamp(0.45 * zoneNearness + 0.25 * zoneDepthCentrality + 0.20 * MathAbs(zf.zoneRelevance) + 0.10 * MathAbs(zf.zoneRetestBreakState), 0.0, 1.0);
      double rejection = 0.0;
      if(zf.zoneSideState > 0.0) rejection = Clamp((lower / range) * (closeLoc), 0.0, 1.0);
      else if(zf.zoneSideState < 0.0) rejection = Clamp((upper / range) * (1.0 - closeLoc), 0.0, 1.0);
      rejection *= zoneContext;
      double acceptance = 0.0;
      if(zf.zoneSideState > 0.0) acceptance = Clamp((1.0 - closeLoc) * (body / range), 0.0, 1.0);
      else if(zf.zoneSideState < 0.0) acceptance = Clamp((closeLoc) * (body / range), 0.0, 1.0);
      acceptance *= zoneContext;
      double indecision = Clamp((1.0 - body / range) * (0.35 + 0.65 * zoneContext), 0.0, 1.0);
      string txt = "";
      color clr = clrSilver;
      if(rejection >= 0.45) { txt = TFLabel(tf) + " Reject"; clr = clrDodgerBlue; }
      else if(acceptance >= 0.45) { txt = TFLabel(tf) + " Accept"; clr = clrOrangeRed; }
      else if(indecision >= 0.60) { txt = TFLabel(tf) + " Indecision"; clr = clrSlateBlue; }
      if(StringLen(txt) > 0)
         DrawCandleFlagLabel(pfx + "CF_0", iTime(symbol, tf, 1), high1, txt, clr);
   }
}

void UpdateStructureVisualization(const string symbol)
{
   if(!(DrawSwingStructureOnChart || DrawSwingZonesOnChart || DrawZoneCandleFlagsOnChart))
      return;
   ENUM_TIMEFRAMES tfExec = (TF_EXEC==PERIOD_CURRENT ? BaseTF : TF_EXEC);
   UpdateStructureVisualizationForTF(symbol, tfExec);
   UpdateStructureVisualizationForTF(symbol, TF_MID);
   UpdateStructureVisualizationForTF(symbol, TF_LONG);
   if(TF_STRUCT_EXT > 0)
      UpdateStructureVisualizationForTF(symbol, TF_STRUCT_EXT);
}



string BuildUnifiedDQNFile(const string symbol)
{
   return MQLInfoString(MQL_PROGRAM_NAME) + "_DQN_MAIN_" + symbol + ".dat";
}
string BuildRegimeDQNFile(const string symbol,const int regime)
{
   return MQLInfoString(MQL_PROGRAM_NAME) + "_DQN_MAIN_" + symbol + "_R" + IntegerToString(regime) + ".dat";
}
string BuildArchiveEpisodesFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_ARCHIVE_Episodes_" + symbol + ".dat"; }
string BuildArchivePatternsFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_ARCHIVE_Patterns_" + symbol + ".dat"; }
string BuildArchiveRegimesFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_ARCHIVE_Regimes_" + symbol + ".dat"; }
string BuildReplayMainFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Main_" + symbol + ".dat"; }
string BuildReplayDangerFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Danger_" + symbol + ".dat"; }
string BuildReplayDeepFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_DeepBasket_" + symbol + ".dat"; }
string BuildReplayEfficientFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Efficient_" + symbol + ".dat"; }
string BuildReplayRecentFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Recent_" + symbol + ".dat"; }

void WriteIntArraySimple(const int h,const int &arr[])
{
   int n=ArraySize(arr); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) FileWriteInteger(h,arr[i]);
}
void ReadIntArraySimple(const int h,int &arr[])
{
   int n=FileReadInteger(h); if(n<0) n=0;
   ArrayResize(arr,n);
   for(int i=0;i<n;i++) arr[i]=FileReadInteger(h);
}
void WriteLongArraySimple(const int h,const long &arr[])
{
   int n=ArraySize(arr); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) FileWriteLong(h,arr[i]);
}
void ReadLongArraySimple(const int h,long &arr[])
{
   int n=FileReadInteger(h); if(n<0) n=0;
   ArrayResize(arr,n);
   for(int i=0;i<n;i++) arr[i]=FileReadLong(h);
}
void WriteBarSpanRef(const int h,const BarSpanRef &r)
{
   FileWriteInteger(h,(int)r.tf);
   FileWriteLong(h,(long)r.startTime);
   FileWriteLong(h,(long)r.endTime);
   FileWriteInteger(h,r.startIndex);
   FileWriteInteger(h,r.endIndex);
}
void ReadBarSpanRef(const int h,BarSpanRef &r)
{
   r.tf=(ENUM_TIMEFRAMES)FileReadInteger(h);
   r.startTime=(datetime)FileReadLong(h);
   r.endTime=(datetime)FileReadLong(h);
   r.startIndex=FileReadInteger(h);
   r.endIndex=FileReadInteger(h);
}
void WriteReplayItemEx(const int h,const ReplayItem &it)
{
   FileWriteInteger(h,it.symIdx); FileWriteInteger(h,it.regime); FileWriteInteger(h,it.action);
   FileWriteDouble(h,it.reward); FileWriteInteger(h,(int)it.done);
   WriteDoubleArray(h,it.state); WriteDoubleArray(h,it.nextState);
   FileWriteDouble(h,it.priority);
   FileWriteInteger(h,it.replayBankType);
   FileWriteLong(h,it.episodeId); FileWriteLong(h,it.patternId); FileWriteLong(h,it.regimeId); FileWriteLong(h,it.periodId);
   FileWriteInteger(h,it.basketStateClass); FileWriteInteger(h,it.addDepthClass); FileWriteInteger(h,it.dangerClass);
   FileWriteInteger(h,it.zoneContextClass); FileWriteInteger(h,it.structureContextClass); FileWriteInteger(h,it.liquidityClass);
   FileWriteLong(h,(long)it.eventTime);
   FileWriteDouble(h,it.painSeverity);
   FileWriteDouble(h,it.recurrenceScore);
   FileWriteDouble(h,it.regimeBreakScore);
   WriteDoubleArray(h,it.macroSig);
   WriteDoubleArray(h,it.microSig);
}
void ReadReplayItemEx(const int h,ReplayItem &it)
{
   it.symIdx=FileReadInteger(h); it.regime=FileReadInteger(h); it.action=FileReadInteger(h);
   it.reward=FileReadDouble(h); it.done=(bool)FileReadInteger(h);
   ReadDoubleArray(h,it.state); ReadDoubleArray(h,it.nextState);
   it.priority=FileReadDouble(h);
   it.replayBankType=FileReadInteger(h);
   it.episodeId=FileReadLong(h); it.patternId=FileReadLong(h); it.regimeId=FileReadLong(h); it.periodId=FileReadLong(h);
   it.basketStateClass=FileReadInteger(h); it.addDepthClass=FileReadInteger(h); it.dangerClass=FileReadInteger(h);
   it.zoneContextClass=FileReadInteger(h); it.structureContextClass=FileReadInteger(h); it.liquidityClass=FileReadInteger(h);
   it.eventTime=(datetime)FileReadLong(h);
   if(!FileIsEnding(h))
   {
      it.painSeverity=FileReadDouble(h);
      it.recurrenceScore=FileReadDouble(h);
      it.regimeBreakScore=FileReadDouble(h);
      ReadDoubleArray(h,it.macroSig);
      ReadDoubleArray(h,it.microSig);
   }
   else
   {
      it.painSeverity=0.0; it.recurrenceScore=0.0; it.regimeBreakScore=0.0;
      ArrayResize(it.macroSig,0); ArrayResize(it.microSig,0);
   }
}
bool SaveReplayBankStoreFile(const string filename,const ReplayBankStore &bank)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,2); FileWriteInteger(h,bank.count); FileWriteInteger(h,bank.maxCount);
   int n=ArraySize(bank.items); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) WriteReplayItemEx(h,bank.items[i]);
   FileClose(h); return true;
}
bool LoadReplayBankStoreFile(const string filename,ReplayBankStore &bank)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   bank.count=FileReadInteger(h); bank.maxCount=FileReadInteger(h);
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(bank.items,n);
   for(int i=0;i<n;i++) ReadReplayItemEx(h,bank.items[i]);
   FileClose(h); return true;
}

void BuildMainReplayStoreForSymbol(const int symIdx,ReplayBankStore &bankOut)
{
   ArrayResize(bankOut.items,0);
   bankOut.count=0;
   bankOut.maxCount=ReplayCapacity;
   int n=ArraySize(gReplay);
   for(int i=0;i<n;i++)
   {
      if(gReplay[i].symIdx != symIdx) continue;
      int k=ArraySize(bankOut.items);
      ArrayResize(bankOut.items,k+1);
      bankOut.items[k]=gReplay[i];
   }
   bankOut.count=ArraySize(bankOut.items);
}

bool SaveMainReplayForSymbol(const int symIdx,const string filename)
{
   ReplayBankStore bank;
   BuildMainReplayStoreForSymbol(symIdx, bank);
   return SaveReplayBankStoreFile(filename, bank);
}

bool AppendMainReplayForSymbol(const string filename,const int symIdx)
{
   ReplayBankStore bank;
   ArrayResize(bank.items,0);
   bank.count=0; bank.maxCount=ReplayCapacity;
   if(!LoadReplayBankStoreFile(filename, bank)) return false;

   int n=ArraySize(bank.items);
   for(int i=0;i<n;i++)
   {
      if(bank.items[i].symIdx != symIdx) continue;
      int k=ArraySize(gReplay);
      ArrayResize(gReplay, k+1);
      gReplay[k]=bank.items[i];
   }
   if(ArraySize(gReplay) > ReplayCapacity)
      ArrayRemove(gReplay,0,ArraySize(gReplay)-ReplayCapacity);
   return (n>0);
}
bool SaveDeepBasketReplayStoreFile(const string filename,const DeepBasketReplayStore &bank)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1);
   FileWriteInteger(h,bank.itemCount); FileWriteInteger(h,bank.seqCount); FileWriteInteger(h,bank.maxItems); FileWriteInteger(h,bank.maxSeqs);
   int n=ArraySize(bank.items); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) WriteReplayItemEx(h,bank.items[i]);
   int s=ArraySize(bank.sequences); FileWriteInteger(h,s);
   for(int i=0;i<s;i++){
      FileWriteLong(h,bank.sequences[i].episodeId); FileWriteInteger(h,bank.sequences[i].symIdx);
      FileWriteLong(h,(long)bank.sequences[i].startTime); FileWriteLong(h,(long)bank.sequences[i].endTime);
      FileWriteInteger(h,bank.sequences[i].basketDir); FileWriteInteger(h,bank.sequences[i].addCount); FileWriteInteger(h,bank.sequences[i].maxPositions);
      FileWriteDouble(h,bank.sequences[i].maxDD); FileWriteDouble(h,bank.sequences[i].finalReward);
      WriteIntArraySimple(h,bank.sequences[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool LoadDeepBasketReplayStoreFile(const string filename,DeepBasketReplayStore &bank)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   bank.itemCount=FileReadInteger(h); bank.seqCount=FileReadInteger(h); bank.maxItems=FileReadInteger(h); bank.maxSeqs=FileReadInteger(h);
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(bank.items,n);
   for(int i=0;i<n;i++) ReadReplayItemEx(h,bank.items[i]);
   int s=FileReadInteger(h); if(s<0) s=0; ArrayResize(bank.sequences,s);
   for(int i=0;i<s;i++){
      bank.sequences[i].episodeId=FileReadLong(h); bank.sequences[i].symIdx=FileReadInteger(h);
      bank.sequences[i].startTime=(datetime)FileReadLong(h); bank.sequences[i].endTime=(datetime)FileReadLong(h);
      bank.sequences[i].basketDir=FileReadInteger(h); bank.sequences[i].addCount=FileReadInteger(h); bank.sequences[i].maxPositions=FileReadInteger(h);
      bank.sequences[i].maxDD=FileReadDouble(h); bank.sequences[i].finalReward=FileReadDouble(h);
      ReadIntArraySimple(h,bank.sequences[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool SaveEfficientReplayStoreFile(const string filename,const EfficientReplayStore &bank)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1);
   FileWriteInteger(h,bank.itemCount); FileWriteInteger(h,bank.periodCount); FileWriteInteger(h,bank.maxItems); FileWriteInteger(h,bank.maxPeriods);
   int n=ArraySize(bank.items); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) WriteReplayItemEx(h,bank.items[i]);
   int s=ArraySize(bank.periods); FileWriteInteger(h,s);
   for(int i=0;i<s;i++){
      FileWriteLong(h,bank.periods[i].periodId); FileWriteInteger(h,bank.periods[i].symIdx);
      FileWriteLong(h,(long)bank.periods[i].startTime); FileWriteLong(h,(long)bank.periods[i].endTime);
      FileWriteDouble(h,bank.periods[i].rewardTotal); FileWriteDouble(h,bank.periods[i].rewardEfficiency); FileWriteDouble(h,bank.periods[i].ddMax);
      FileWriteInteger(h,bank.periods[i].oneRoundCount); FileWriteInteger(h,bank.periods[i].addCountTotal);
      FileWriteInteger(h,bank.periods[i].sessionType); FileWriteInteger(h,bank.periods[i].liquidityType); FileWriteInteger(h,bank.periods[i].regimeType); FileWriteInteger(h,bank.periods[i].patternType);
      WriteIntArraySimple(h,bank.periods[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool LoadEfficientReplayStoreFile(const string filename,EfficientReplayStore &bank)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   bank.itemCount=FileReadInteger(h); bank.periodCount=FileReadInteger(h); bank.maxItems=FileReadInteger(h); bank.maxPeriods=FileReadInteger(h);
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(bank.items,n);
   for(int i=0;i<n;i++) ReadReplayItemEx(h,bank.items[i]);
   int s=FileReadInteger(h); if(s<0) s=0; ArrayResize(bank.periods,s);
   for(int i=0;i<s;i++){
      bank.periods[i].periodId=FileReadLong(h); bank.periods[i].symIdx=FileReadInteger(h);
      bank.periods[i].startTime=(datetime)FileReadLong(h); bank.periods[i].endTime=(datetime)FileReadLong(h);
      bank.periods[i].rewardTotal=FileReadDouble(h); bank.periods[i].rewardEfficiency=FileReadDouble(h); bank.periods[i].ddMax=FileReadDouble(h);
      bank.periods[i].oneRoundCount=FileReadInteger(h); bank.periods[i].addCountTotal=FileReadInteger(h);
      bank.periods[i].sessionType=FileReadInteger(h); bank.periods[i].liquidityType=FileReadInteger(h); bank.periods[i].regimeType=FileReadInteger(h); bank.periods[i].patternType=FileReadInteger(h);
      ReadIntArraySimple(h,bank.periods[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool SaveEpisodeMemoryFile(const string filename)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1); int n=ArraySize(gEpisodeMemory); FileWriteInteger(h,n);
   for(int i=0;i<n;i++){
      EpisodeMemory e = gEpisodeMemory[i];
      FileWriteLong(h,e.episodeId); FileWriteString(h,e.symbol); FileWriteInteger(h,e.symIdx);
      FileWriteLong(h,(long)e.startTime); FileWriteLong(h,(long)e.endTime);
      FileWriteInteger(h,e.basketDir); FileWriteInteger(h,e.openPositionsMax); FileWriteInteger(h,e.addCount); FileWriteInteger(h,e.actionsCount);
      FileWriteDouble(h,e.entryPriceFirst); FileWriteDouble(h,e.avgEntryAtWorst); FileWriteDouble(h,e.closePriceFinal);
      FileWriteDouble(h,e.pnlFinal); FileWriteDouble(h,e.rewardTotal); FileWriteDouble(h,e.rewardEfficiency); FileWriteDouble(h,e.maxDrawdownPct); FileWriteDouble(h,e.maxDangerScore); FileWriteDouble(h,e.maxMarginStress);
      FileWriteInteger(h,e.oneRoundTrade); FileWriteInteger(h,e.forcedStopLikeEvent); FileWriteInteger(h,e.inefficientRecovery);
      FileWriteInteger(h,e.sessionType); FileWriteInteger(h,e.liquidityType); FileWriteInteger(h,e.regimeType); FileWriteInteger(h,e.patternType);
      WriteBarSpanRef(h,e.execSpan); WriteBarSpanRef(h,e.midSpan); WriteBarSpanRef(h,e.longSpan); WriteBarSpanRef(h,e.structExtSpan);
   }
   FileClose(h); return true;
}
bool LoadEpisodeMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(gEpisodeMemory,n);
   for(int i=0;i<n;i++){
      gEpisodeMemory[i].episodeId=FileReadLong(h); gEpisodeMemory[i].symbol=FileReadString(h); gEpisodeMemory[i].symIdx=FileReadInteger(h);
      gEpisodeMemory[i].startTime=(datetime)FileReadLong(h); gEpisodeMemory[i].endTime=(datetime)FileReadLong(h);
      gEpisodeMemory[i].basketDir=FileReadInteger(h); gEpisodeMemory[i].openPositionsMax=FileReadInteger(h); gEpisodeMemory[i].addCount=FileReadInteger(h); gEpisodeMemory[i].actionsCount=FileReadInteger(h);
      gEpisodeMemory[i].entryPriceFirst=FileReadDouble(h); gEpisodeMemory[i].avgEntryAtWorst=FileReadDouble(h); gEpisodeMemory[i].closePriceFinal=FileReadDouble(h);
      gEpisodeMemory[i].pnlFinal=FileReadDouble(h); gEpisodeMemory[i].rewardTotal=FileReadDouble(h); gEpisodeMemory[i].rewardEfficiency=FileReadDouble(h); gEpisodeMemory[i].maxDrawdownPct=FileReadDouble(h); gEpisodeMemory[i].maxDangerScore=FileReadDouble(h); gEpisodeMemory[i].maxMarginStress=FileReadDouble(h);
      gEpisodeMemory[i].oneRoundTrade=FileReadInteger(h); gEpisodeMemory[i].forcedStopLikeEvent=FileReadInteger(h); gEpisodeMemory[i].inefficientRecovery=FileReadInteger(h);
      gEpisodeMemory[i].sessionType=FileReadInteger(h); gEpisodeMemory[i].liquidityType=FileReadInteger(h); gEpisodeMemory[i].regimeType=FileReadInteger(h); gEpisodeMemory[i].patternType=FileReadInteger(h);
      ReadBarSpanRef(h,gEpisodeMemory[i].execSpan); ReadBarSpanRef(h,gEpisodeMemory[i].midSpan); ReadBarSpanRef(h,gEpisodeMemory[i].longSpan); ReadBarSpanRef(h,gEpisodeMemory[i].structExtSpan);
   }
   FileClose(h); return true;
}
bool SavePatternMemoryFile(const string filename)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1); int n=ArraySize(gPatternMemory); FileWriteInteger(h,n);
   for(int i=0;i<n;i++){
      FileWriteLong(h,gPatternMemory[i].patternId); FileWriteString(h,gPatternMemory[i].symbol); FileWriteInteger(h,gPatternMemory[i].symIdx);
      FileWriteInteger(h,gPatternMemory[i].patternType); FileWriteInteger(h,gPatternMemory[i].strengthClass); FileWriteInteger(h,gPatternMemory[i].volatilityClass); FileWriteInteger(h,gPatternMemory[i].liquidityClass);
      FileWriteLong(h,(long)gPatternMemory[i].startTime); FileWriteLong(h,(long)gPatternMemory[i].endTime);
      FileWriteDouble(h,gPatternMemory[i].rewardEfficiencyMean); FileWriteDouble(h,gPatternMemory[i].ddMean); FileWriteDouble(h,gPatternMemory[i].addMean);
      WriteBarSpanRef(h,gPatternMemory[i].execSpan); WriteBarSpanRef(h,gPatternMemory[i].midSpan); WriteBarSpanRef(h,gPatternMemory[i].longSpan);
      WriteLongArraySimple(h,gPatternMemory[i].linkedEpisodeIds);
   }
   FileClose(h); return true;
}
bool LoadPatternMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(gPatternMemory,n);
   for(int i=0;i<n;i++){
      gPatternMemory[i].patternId=FileReadLong(h); gPatternMemory[i].symbol=FileReadString(h); gPatternMemory[i].symIdx=FileReadInteger(h);
      gPatternMemory[i].patternType=FileReadInteger(h); gPatternMemory[i].strengthClass=FileReadInteger(h); gPatternMemory[i].volatilityClass=FileReadInteger(h); gPatternMemory[i].liquidityClass=FileReadInteger(h);
      gPatternMemory[i].startTime=(datetime)FileReadLong(h); gPatternMemory[i].endTime=(datetime)FileReadLong(h);
      gPatternMemory[i].rewardEfficiencyMean=FileReadDouble(h); gPatternMemory[i].ddMean=FileReadDouble(h); gPatternMemory[i].addMean=FileReadDouble(h);
      ReadBarSpanRef(h,gPatternMemory[i].execSpan); ReadBarSpanRef(h,gPatternMemory[i].midSpan); ReadBarSpanRef(h,gPatternMemory[i].longSpan);
      ReadLongArraySimple(h,gPatternMemory[i].linkedEpisodeIds);
   }
   FileClose(h); return true;
}
bool SaveRegimeEventMemoryFile(const string filename)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1); int n=ArraySize(gRegimeEventMemory); FileWriteInteger(h,n);
   for(int i=0;i<n;i++){
      FileWriteLong(h,gRegimeEventMemory[i].regimeId); FileWriteString(h,gRegimeEventMemory[i].symbol); FileWriteInteger(h,gRegimeEventMemory[i].symIdx);
      FileWriteInteger(h,gRegimeEventMemory[i].regimeType); FileWriteInteger(h,gRegimeEventMemory[i].eventType);
      FileWriteLong(h,(long)gRegimeEventMemory[i].startTime); FileWriteLong(h,(long)gRegimeEventMemory[i].endTime);
      FileWriteDouble(h,gRegimeEventMemory[i].avgVol); FileWriteDouble(h,gRegimeEventMemory[i].avgSpread); FileWriteDouble(h,gRegimeEventMemory[i].avgADX); FileWriteDouble(h,gRegimeEventMemory[i].avgRewardEfficiency);
      WriteBarSpanRef(h,gRegimeEventMemory[i].execSpan); WriteBarSpanRef(h,gRegimeEventMemory[i].midSpan); WriteBarSpanRef(h,gRegimeEventMemory[i].longSpan);
      WriteLongArraySimple(h,gRegimeEventMemory[i].linkedEpisodeIds); WriteLongArraySimple(h,gRegimeEventMemory[i].linkedPatternIds);
   }
   FileClose(h); return true;
}
bool LoadRegimeEventMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(gRegimeEventMemory,n);
   for(int i=0;i<n;i++){
      gRegimeEventMemory[i].regimeId=FileReadLong(h); gRegimeEventMemory[i].symbol=FileReadString(h); gRegimeEventMemory[i].symIdx=FileReadInteger(h);
      gRegimeEventMemory[i].regimeType=FileReadInteger(h); gRegimeEventMemory[i].eventType=FileReadInteger(h);
      gRegimeEventMemory[i].startTime=(datetime)FileReadLong(h); gRegimeEventMemory[i].endTime=(datetime)FileReadLong(h);
      gRegimeEventMemory[i].avgVol=FileReadDouble(h); gRegimeEventMemory[i].avgSpread=FileReadDouble(h); gRegimeEventMemory[i].avgADX=FileReadDouble(h); gRegimeEventMemory[i].avgRewardEfficiency=FileReadDouble(h);
      ReadBarSpanRef(h,gRegimeEventMemory[i].execSpan); ReadBarSpanRef(h,gRegimeEventMemory[i].midSpan); ReadBarSpanRef(h,gRegimeEventMemory[i].longSpan);
      ReadLongArraySimple(h,gRegimeEventMemory[i].linkedEpisodeIds); ReadLongArraySimple(h,gRegimeEventMemory[i].linkedPatternIds);
   }
   FileClose(h); return true;
}
bool SaveUnifiedPersistenceForSymbol(const int symIdx)
{
   bool ok=true;
   // Unified persistence is always the active DQN persistence path.
   if(UseRegimeBank)
   {
      for(int r=0;r<REGIME_COUNT;r++)
         ok = SaveDQNForSymbol(symIdx,r,BuildRegimeDQNFile(gSymbols[symIdx], r)) && ok;
      ok = SaveDQNForSymbol(symIdx,0,BuildUnifiedDQNFile(gSymbols[symIdx])) && ok;
   }
   else
      ok = SaveDQNForSymbol(symIdx,0,BuildUnifiedDQNFile(gSymbols[symIdx])) && ok;
   if(SaveArchiveMemory)
   {
      ok = SaveEpisodeMemoryFile(BuildArchiveEpisodesFile(gSymbols[symIdx])) && ok;
      ok = SavePatternMemoryFile(BuildArchivePatternsFile(gSymbols[symIdx])) && ok;
      ok = SaveRegimeEventMemoryFile(BuildArchiveRegimesFile(gSymbols[symIdx])) && ok;
   }
   if(SaveReplayBanks)
   {
      if(SaveMainReplayBank)
         ok = SaveMainReplayForSymbol(symIdx, BuildReplayMainFile(gSymbols[symIdx])) && ok;
      ok = SaveReplayBankStoreFile(BuildReplayDangerFile(gSymbols[symIdx]), gDangerReplayBank) && ok;
      ok = SaveDeepBasketReplayStoreFile(BuildReplayDeepFile(gSymbols[symIdx]), gDeepBasketReplayBank) && ok;
      ok = SaveEfficientReplayStoreFile(BuildReplayEfficientFile(gSymbols[symIdx]), gEfficientReplayBank) && ok;
      if(SaveRecentReplayBank)
         ok = SaveReplayBankStoreFile(BuildReplayRecentFile(gSymbols[symIdx]), gRecentReplayBank) && ok;
   }
   return ok;
}
bool LoadUnifiedPersistenceForSymbol(const int symIdx,int inDim)
{
   bool loaded=false;
   bool anyDqnLoaded=false;

   if(UseRegimeBank)
   {
      bool haveRegime0=false;
      for(int r=0;r<REGIME_COUNT;r++)
      {
         string rf=BuildRegimeDQNFile(gSymbols[symIdx], r);
         bool okR=LoadDQNForSymbol(symIdx,r,rf);
         if(!okR && r==0)
            okR=LoadDQNForSymbol(symIdx,0,BuildUnifiedDQNFile(gSymbols[symIdx]));
         if(okR && gDQN[symIdx][r].input_dim==inDim)
         {
            anyDqnLoaded=true;
            loaded=true;
            if(r==0) haveRegime0=true;
         }
         else
         {
            if(r==0)
            {
               InitOrRandomizeDQN(symIdx,0,inDim);
            }
            else if(haveRegime0)
            {
               gDQN[symIdx][r]=gDQN[symIdx][0];
               gTargetDQN[symIdx][r]=gTargetDQN[symIdx][0];
            }
            else
            {
               InitOrRandomizeDQN(symIdx,r,inDim);
            }
         }
      }
      if(anyDqnLoaded)
      {
         if(StringLen(gLoadedDQNMetaNote[symIdx])>0)
            Print("Loaded DQN metadata [", gSymbols[symIdx], "]: ", gLoadedDQNMetaNote[symIdx]);
         if(gLoadedDQNFastModeMeta[symIdx] && !FastTrainingMode)
            Print("Loaded DQN note [", gSymbols[symIdx], "]: model was last saved from fast-training mode; full-feature inference is load-compatible, but a full-feature fine-tune is recommended before live deployment.");
      }
   }
   else
   {
      string fname=BuildUnifiedDQNFile(gSymbols[symIdx]);
      bool dqnLoaded=LoadDQNForSymbol(symIdx,0,fname);
      if(!dqnLoaded || gDQN[symIdx][0].input_dim!=inDim)
         InitOrRandomizeDQN(symIdx,0,inDim);
      else
      {
         loaded=true;
         anyDqnLoaded=true;
         if(StringLen(gLoadedDQNMetaNote[symIdx])>0)
            Print("Loaded DQN metadata [", gSymbols[symIdx], "]: ", gLoadedDQNMetaNote[symIdx]);
         if(gLoadedDQNFastModeMeta[symIdx] && !FastTrainingMode)
            Print("Loaded DQN note [", gSymbols[symIdx], "]: model was last saved from fast-training mode; full-feature inference is load-compatible, but a full-feature fine-tune is recommended before live deployment.");
      }
      for(int r=1;r<REGIME_COUNT;r++)
      {
         gDQN[symIdx][r]=gDQN[symIdx][0];
         gTargetDQN[symIdx][r]=gTargetDQN[symIdx][0];
      }
   }

   if(SaveArchiveMemory)
   {
      loaded = LoadEpisodeMemoryFile(BuildArchiveEpisodesFile(gSymbols[symIdx])) || loaded;
      loaded = LoadPatternMemoryFile(BuildArchivePatternsFile(gSymbols[symIdx])) || loaded;
      loaded = LoadRegimeEventMemoryFile(BuildArchiveRegimesFile(gSymbols[symIdx])) || loaded;
   }
   if(SaveReplayBanks)
   {
      if(SaveMainReplayBank)
         loaded = AppendMainReplayForSymbol(BuildReplayMainFile(gSymbols[symIdx]), symIdx) || loaded;
      loaded = LoadReplayBankStoreFile(BuildReplayDangerFile(gSymbols[symIdx]), gDangerReplayBank) || loaded;
      loaded = LoadDeepBasketReplayStoreFile(BuildReplayDeepFile(gSymbols[symIdx]), gDeepBasketReplayBank) || loaded;
      loaded = LoadEfficientReplayStoreFile(BuildReplayEfficientFile(gSymbols[symIdx]), gEfficientReplayBank) || loaded;
      if(SaveRecentReplayBank)
         loaded = LoadReplayBankStoreFile(BuildReplayRecentFile(gSymbols[symIdx]), gRecentReplayBank) || loaded;
   }
   return loaded;
}

int OnInit()
{
   for(int i=0;i<MAX_SYMBOLS;i++)
   {
      gZScoreLastClosedBarTF[i]=0;
      gZScoreLastValue[i]=0.0;
      gZScoreLastAbs[i]=0.0;
      gZScoreExtremeNow[i]=false;
      gZScorePauseTrading[i]=false;
      gZScoreQuietBars[i]=0;
      gZScoreLastExtremeBarTime[i]=0;
   }

   InitPerfCaches();
   trade.SetDeviationInPoints(Slippage);
   MathSrand((int)GetTickCount());

   bool loadedPersistentMemory = false;

   gT1Enter=T1_CautionEnter;
   gT1Exit =T1_CautionExit;
   gT2Enter=T2_DangerEnter;
   gT2Exit =T2_DangerExit;

   gSymbolCount=1;
   gSymbols[0]=_Symbol;
   gMagics[0]=InpMagic;

   PendingInit();
   ArrayResize(gReplay,0);
   ArrayResize(gQMem,0);
   ArrayResize(gBasketHistory,0);
   ArrayResize(gTickTrace,0);
   ArrayResize(gDDEvents,0);

   gDDEventActive=false;
   gDDEventTriggerTime=0;
   gDDEventTriggerDD=0.0;
   gDDEventHard=false;
   gDDEventStartBasketIndex=-1;
   gDDEventStartTickIndex=-1;
   gDDEventPostBasketCount=0;

   for(int i=0;i<gSymbolCount;i++)
   {
      gGridLastBar[i]=0;
      gGridStepCache[i]=0.0;
      gGridActiveChannel[i]=0;
      gGridActiveStep[i]=0.0;

      gExtremeDDStart[i]=0;
      gExtremeDDArmed[i]=false;

      gPositionsCount[i]=0;
      gTrades[i].Clear();
      gTradeLots[i].Clear();
      gFirstTradeTime[i]=0;

      gLastBarTime[i]=0;
      gLastReplayDecisionBarTime[i]=0;
      gReplayDecisionBarCounter[i]=0;
      ResetDenseBasketHealthTracker(i);
      gLoadedDQNFastModeMeta[i]=false;
      gLoadedDQNMetaNote[i]="";
      gDecisionCtxBarTime[i]=0;
      gDecisionCtxHourBucket[i]=0;
      gDecisionCtxRegimeType[i]=0;
      gDecisionCtxPatternType[i]=0;
      gDecisionCtxLiquidityType[i]=0;
      gDecisionCtxSessionType[i]=0;
      gDecisionCtxAtrRatio[i]=0.0;
      gDecisionCtxValid[i]=false;
      gFastQMemLastRefreshTime[i]=0;
      gFastQMemLastDecisionBarTime[i]=0;
      gFastQMemDecisionBarCounter[i]=0;
      gFastQMemCachedRegime[i]=-1000;
      gFastQMemCachedValid[i]=false;
      gFastQMemCachedConf[i]=0.0;
      for(int a=0;a<3;a++)
         gFastQMemCachedQ[i][a]=0.0;

      ResetDecisionSupportContext(gDecisionSupportCache[i]);

      gD1LastBarTime[i]=0;
      gD1DistCache[i]=0.0;
      gD1SlopeCache[i]=0.0;
      gD1SideDurCache[i]=0.0;

      gATRLastBarTime[i]=0;
      gATRratioCache[i]=1.0;

      hRSI_Base[i]  = iRSI(gSymbols[i], BaseTF, RSI_Period, PRICE_CLOSE);
      hCCI_Base[i]  = iCCI(gSymbols[i], BaseTF, CCI_Period, PRICE_TYPICAL);
      hMACD_Base[i] = iMACD(gSymbols[i], BaseTF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
      hEMA_Base[i]  = iMA(gSymbols[i], BaseTF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
      hRVI_Base[i]  = iRVI(gSymbols[i], BaseTF, RVI_Period);
      hATRfast_Base[i] = iATR(gSymbols[i], BaseTF, ATR_FastPeriod);
      hATRslow_Base[i] = iATR(gSymbols[i], BaseTF, ATR_SlowPeriod);

      gIndLastBarTime[i]=0;
      gRSI_Base[i]=50.0;
      gCCI_Base[i]=0.0;
      gMACD_BaseMain[i]=0.0;
      gEMA_BaseVal[i]=0.0;
      gRVI_BaseMain[i]=0.0;
      gATRfast_BaseVal[i]=0.0;
      gATRslow_BaseVal[i]=0.0;

      hRSI_H1[i]=INVALID_HANDLE; hMACD_H1[i]=INVALID_HANDLE; hEMA_H1[i]=INVALID_HANDLE; hRVI_H1[i]=INVALID_HANDLE;
      hATRfast_H1[i]=INVALID_HANDLE; hATRslow_H1[i]=INVALID_HANDLE;
      gH1LastBar[i]=0; gRSI_H1v[i]=50.0; gMACD_H1v[i]=0.0; gEMA_H1v[i]=0.0; gRVI_H1v[i]=0.0; gATRratio_H1[i]=1.0;
      
      gLastTradeOpenTime[i]=TimeCurrent();
      gForcedEntryActive[i]=false;
      gForcedEntryArmTime[i]=0;
      gForcedEntryDeadline[i]=0;
       
      if(UseH1Features)
      {
         hRSI_H1[i]     = iRSI(gSymbols[i], H1_TF, RSI_Period, PRICE_CLOSE);
         hMACD_H1[i]    = iMACD(gSymbols[i], H1_TF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
         hEMA_H1[i]     = iMA(gSymbols[i], H1_TF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
         hRVI_H1[i]     = iRVI(gSymbols[i], H1_TF, RVI_Period);
         hATRfast_H1[i] = iATR(gSymbols[i], H1_TF, ATR_FastPeriod);
         hATRslow_H1[i] = iATR(gSymbols[i], H1_TF, ATR_SlowPeriod);
      }

      hRSI_H4[i]=INVALID_HANDLE; hMACD_H4[i]=INVALID_HANDLE; hEMA_H4[i]=INVALID_HANDLE; hRVI_H4[i]=INVALID_HANDLE;
      hATRfast_H4[i]=INVALID_HANDLE; hATRslow_H4[i]=INVALID_HANDLE;
      gH4LastBar[i]=0; gRSI_H4v[i]=50.0; gMACD_H4v[i]=0.0; gEMA_H4v[i]=0.0; gRVI_H4v[i]=0.0; gATRratio_H4[i]=1.0;

      if(UseH4Features)
      {
         hRSI_H4[i]     = iRSI(gSymbols[i], H4_TF, RSI_Period, PRICE_CLOSE);
         hMACD_H4[i]    = iMACD(gSymbols[i], H4_TF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
         hEMA_H4[i]     = iMA(gSymbols[i], H4_TF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
         hRVI_H4[i]     = iRVI(gSymbols[i], H4_TF, RVI_Period);
         hATRfast_H4[i] = iATR(gSymbols[i], H4_TF, ATR_FastPeriod);
         hATRslow_H4[i] = iATR(gSymbols[i], H4_TF, ATR_SlowPeriod);
      }

      double tmpState[];
      BuildState(gSymbols[i], i, 0, gTrades[i], tmpState);
      int inDim = ArraySize(tmpState);
      if(inDim<=0) inDim=8;

      bool unifiedLoaded = LoadUnifiedPersistenceForSymbol(i,inDim);
      if(unifiedLoaded)
         loadedPersistentMemory = true;
      else
      {
         // Unified persistence is the only DQN persistence path now.
         // If no persisted model exists yet, initialize a unified model
         // in regime slot 0 and mirror it into the compatibility regime slots.
         InitOrRandomizeDQN(i,0,inDim);
         for(int r=1;r<REGIME_COUNT;r++)
         {
            gDQN[i][r] = gDQN[i][0];
            gTargetDQN[i][r] = gTargetDQN[i][0];
         }
      }

      gMode[i]=MODE_NORMAL;
      gModeLastChange[i]=0;
      gPDanger[i]=0.0;
      gLRScale[i]=1.0;

      gFPLastBar[i]=0;
      for(int k=0;k<6;k++) { gFPCache[i][k]=0.0; gFPPrev[i][k]=0.0; }
      gFPPrevValid[i]=false;

      gMiniLastBar[i]=0;
      gMiniValid[i]=false;
      gMiniPrevValid[i]=false;
      for(int k=0;k<MINI_DIM;k++)
      {
         gMiniCache[i][k]=0.0;
         gMiniPrev[i][k]=0.0;
      }

      gBadDDStart[i]=0;
      gBadDDConfirmed[i]=false;
      gBadLabel[i]=-1;
      gBadBasketDir[i]=0;
      gBadTrendDir[i]=0;
      gBadAtrRatio[i]=1.0;

      gBadHaveBasketFP[i]=false;
      gBadHaveQStart[i]=false;
      gBadHavePreFP[i]=false;
      gBadHaveEndFP[i]=false;

      gBadHaveMiniStart[i]=false;
      gBadHaveMiniBasket[i]=false;
      gBadHaveMiniPre[i]=false;
      gBadHaveMiniEnd[i]=false;

      for(int k=0;k<6;k++)
      {
         gBadFPStart[i][k]=0.0;
         gBadFPBasketOpen[i][k]=0.0;
         gBadFPPre[i][k]=0.0;
         gBadFPEnd[i][k]=0.0;
      }
      for(int a=0;a<8;a++) gBadQStart[i][a]=0.0;

      for(int k=0;k<MINI_DIM;k++)
      {
         gBadMiniStart[i][k]=0.0;
         gBadMiniBasketOpen[i][k]=0.0;
         gBadMiniPre[i][k]=0.0;
         gBadMiniEnd[i][k]=0.0;
      }

      gLastForcedStopLossMoney[i]=0.0;
      gLastForcedStopTime[i]=0;
      gLastForcedStopPendingLearn[i]=false;

      gForcedStopHaveFP[i]=false;
      gForcedStopHaveMini[i]=false;
      gForcedStopHaveQ[i]=false;

      gForcedStopBasketDir[i]=0;
      gForcedStopTrendDir[i]=0;
      gForcedStopAtrRatio[i]=1.0;
      gForcedStopLabel[i]=-1;

      for(int k=0;k<6;k++)
         gForcedStopFP[i][k]=0.0;

      for(int k=0;k<MINI_DIM;k++)
         gForcedStopMini[i][k]=0.0;

      for(int a=0;a<8;a++)
         gForcedStopQ[i][a]=0.0;

      gCachedSymbolPnL[i]=0.0;
      gHasPositionTypeCache[i]=false;
      gBasketDirCache[i]=0;
      gBasketAvgPriceCache[i]=0.0;
      gBasketAvgValid[i]=false;
   }

   if(UseDangerBrain)
   {
      string memFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DangerMem_" + _Symbol + ".dat";
      bool loaded=false;
      if(SaveDangerMemory) loaded = LoadDangerMemoryFile(memFile);
      if(!loaded) SeedTestProtos();
      else loadedPersistentMemory = true;

      PruneProtosIfNeeded();
      EventSetTimer(3);
   }

   if(UseQMemory && SaveQMemory)
   {
      string qmemFile = MQLInfoString(MQL_PROGRAM_NAME) + "_QMem_" + _Symbol + ".dat";
      if(LoadQMemoryFile(qmemFile))
         loadedPersistentMemory = true;
   }

   if(UseDDEventMemory && SaveDDEventMemory)
   {
      string ddFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DDEvents_" + _Symbol + ".dat";
      if(LoadDDEventMemoryFile(ddFile))
         loadedPersistentMemory = true;
   }

   gEAStartEquity = (EquityBudget>1e-9 ? EquityBudget : AccountInfoDouble(ACCOUNT_BALANCE));
   if(gEAStartEquity<=1e-9) gEAStartEquity=1000.0;

   gEAClosedProfit=0.0;
   gEquityLossStopResumeTime=0;
   gProfitPauseResumeTime=0;
   gProfitCycleClosedProfit=0.0;
   
   // virtual-budget model baseline stays unchanged
   maxEquity=gEAStartEquity;
   gTickEquityBaseline=gEAStartEquity;
   gTickBalanceBaseline=gEAStartEquity;
   gRewardBaselineTick=0;
   
   // separate real account/watchdog peak for equity stop only
   gAccountEquityStopPeak = GetWatchedAccountEquity();
   if(gAccountEquityStopPeak <= 1e-9)
      gAccountEquityStopPeak = AccountInfoDouble(ACCOUNT_BALANCE);
   if(gAccountEquityStopPeak <= 1e-9)
      gAccountEquityStopPeak = 1000.0;

   isTraining=TrainingMode;
   tickCounter=0;
   totalReward=0.0;
   episodeCount=0;
   gTargetSyncCounter=0;

   if(isTraining)
   {
      if(ResumeLowEpsilon && loadedPersistentMemory)
         currentEpsilon = MathMax(MinExplorationRate, ExplorationRate * LoadedEpsilonFactor);
      else
         currentEpsilon = ExplorationRate;
   }
   else
   {
      currentEpsilon = MinExplorationRate;
   }

   for(int i=0;i<MAX_SYMBOLS;i++)
   {
      gActiveBasketEpisodes[i].active=false;
      gActiveBasketEpisodes[i].episodeId=0;
      gActiveBasketEpisodes[i].symIdx=i;
      ArrayResize(gActiveBasketEpisodes[i].replayItemIndexes,0);

      gActiveEfficientPeriods[i].active=false;
      gActiveEfficientPeriods[i].periodId=0;
      gActiveEfficientPeriods[i].symIdx=i;
      ArrayResize(gActiveEfficientPeriods[i].replayItemIndexes,0);
   }

   SyncAllTargetNetworks();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{

   if(UseDangerBrain)
   {
      EventKillTimer();
      if(PruneOnDeinit) PruneProtosIfNeeded();
      if(SaveDangerMemory)
      {
         string memFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DangerMem_" + _Symbol + ".dat";
         SaveDangerMemoryFile(memFile);
      }
   }

   if(UseQMemory && SaveQMemory)
   {
      string qmemFile = MQLInfoString(MQL_PROGRAM_NAME) + "_QMem_" + _Symbol + ".dat";
      SaveQMemoryFile(qmemFile);
   }

   if(UseDDEventMemory && SaveDDEventMemory)
   {
      if(gDDEventActive)
         FinalizeActiveDDEvent();

      string ddFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DDEvents_" + _Symbol + ".dat";
      SaveDDEventMemoryFile(ddFile);
   }

   if(UseDQN && SaveQTable)
   {
      for(int i=0;i<gSymbolCount;i++)
         SaveUnifiedPersistenceForSymbol(i);
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      if(hRSI_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hRSI_Base[i]);
      if(hCCI_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hCCI_Base[i]);
      if(hMACD_Base[i]!=INVALID_HANDLE) IndicatorRelease(hMACD_Base[i]);
      if(hEMA_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hEMA_Base[i]);
      if(hRVI_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hRVI_Base[i]);
      if(hATRfast_Base[i]!=INVALID_HANDLE) IndicatorRelease(hATRfast_Base[i]);
      if(hATRslow_Base[i]!=INVALID_HANDLE) IndicatorRelease(hATRslow_Base[i]);

      if(hRSI_H1[i]!=INVALID_HANDLE)     IndicatorRelease(hRSI_H1[i]);
      if(hMACD_H1[i]!=INVALID_HANDLE)    IndicatorRelease(hMACD_H1[i]);
      if(hEMA_H1[i]!=INVALID_HANDLE)     IndicatorRelease(hEMA_H1[i]);
      if(hRVI_H1[i]!=INVALID_HANDLE)     IndicatorRelease(hRVI_H1[i]);
      if(hATRfast_H1[i]!=INVALID_HANDLE) IndicatorRelease(hATRfast_H1[i]);
      if(hATRslow_H1[i]!=INVALID_HANDLE) IndicatorRelease(hATRslow_H1[i]);

      if(hRSI_H4[i]!=INVALID_HANDLE)     IndicatorRelease(hRSI_H4[i]);
      if(hMACD_H4[i]!=INVALID_HANDLE)    IndicatorRelease(hMACD_H4[i]);
      if(hEMA_H4[i]!=INVALID_HANDLE)     IndicatorRelease(hEMA_H4[i]);
      if(hRVI_H4[i]!=INVALID_HANDLE)     IndicatorRelease(hRVI_H4[i]);
      if(hATRfast_H4[i]!=INVALID_HANDLE) IndicatorRelease(hATRfast_H4[i]);
      if(hATRslow_H4[i]!=INVALID_HANDLE) IndicatorRelease(hATRslow_H4[i]);
   }

   if(DrawZonesOnChart)
   {
      int total=ArraySize(zones);
      for(int i=0;i<total;i++)
      {
         ObjectDelete(0,zones[i].name);
         ObjectDelete(0,zones[i].name+"_lbl");
      }
   }
   
   ObjectDelete(0,gGridPanelName);
}

void OnTimer()
{
   if(!UseDangerBrain) return;

   datetime now=TimeCurrent();
   for(int i=0;i<gSymbolCount;i++)
   {
      string sym=gSymbols[i];
      datetime bt=iTime(sym, FingerprintTF, 0);
      if(bt==0) continue;
      if(bt==gFPLastBar[i]) continue;

      gFPLastBar[i]=bt;

      double f_now[];
      if(!ComputeFingerprint(sym, FingerprintTF, FingerprintBars, f_now)) continue;

      for(int k=0;k<6;k++) gFPPrev[i][k]=gFPCache[i][k];
      gFPPrevValid[i]=true;

      for(int k=0;k<6;k++) gFPCache[i][k]=f_now[k];

      double pd=0.0;
      int bestDangerIdx = DangerPredict(f_now, i, pd);
      gPDanger[i]=pd;

      UpdateMode(i, pd, now);

      if(bestDangerIdx>=0)
      {
         double exposure=ComputeExposureScore(i);
         if(pd>gT2Enter && exposure<0.05)
            ProtoScoreBump(bestDangerIdx, -0.02);
      }
   }
}

void OnTick()
{
   double equity=GetEAEquity();
   if(equity>maxEquity) maxEquity=equity;

   double watchedEq = GetWatchedAccountEquity();
   if(watchedEq > gAccountEquityStopPeak)
      gAccountEquityStopPeak = watchedEq;

   RefreshRewardTickBaseline();
   ResetProfitPauseIfExpired();
   ResetEquityLossStopCooldownIfExpired();
   
   if(HandleEquityLossStop())
   {
      StartEquityLossStopCooldown();
      CountOpenPositions();
      return;
   }

   if(CheckProfitPauseTrigger())
   {
      CloseAllPositions();
      CountOpenPositions();
      StartProfitPause();
      return;
   }

   if(ProfitPauseActive())
      return;

   if(UseEquityStop && CheckEquityStop())
   {
      CloseAllPositions();
      CountOpenPositions();

      for(int i=0;i<gSymbolCount;i++)
      {
         gFirstTradeTime[i]=0;
         gLastTradeOpenTime[i]=TimeCurrent();
         ResetForcedEntryState(i);
      }

      // reset only the real account-equity stop peak
      gAccountEquityStopPeak = GetWatchedAccountEquity();
      if(gAccountEquityStopPeak <= 1e-9)
         gAccountEquityStopPeak = AccountInfoDouble(ACCOUNT_BALANCE);

      return;
   }

   CountOpenPositions();

   if(UseDangerBrain && gFPLastBar[0]==0)
      OnTimer();

   datetime now = TimeCurrent();

   static datetime s_lastPendingExpireCheck = 0;
   bool runPendingExpire = false;
   if(UsePendingTransitions && ArraySize(gPending) > 0)
   {
      if(now != s_lastPendingExpireCheck)
      {
         s_lastPendingExpireCheck = now;
         runPendingExpire = true;
      }
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      string sym = gSymbols[i];

      UpdateBaseIndicatorsIfNewBar(i);
      UpdateH1IfNewBar(i);
      UpdateH4IfNewBar(i);
      UpdateMultiChannelGridStats(i);
      GetAtrRatioCached_Base(sym);
      UpdateZScoreRiskGuardForSymbol(sym, i);

      datetime lastClosed = iTime(sym, BaseTF, 1);
      bool isNewClosedBar = (lastClosed > 0 && lastClosed != gLastBarTime[i]);

      if(isNewClosedBar)
      {
         gLastBarTime[i] = lastClosed;

         DetectZones(sym);
         UpdateZones(sym);
         UpdateMiniStateCache(i);
         if(!MQLInfoInteger(MQL_TESTER)) UpdateStructureVisualization(sym);
         MaybePrintReplayDiagnostics(i);
      }
   }

   if(runPendingExpire)
      PendingExpireOld(now);

   if(CheckCCIExit())
      return;

   TrailingStop();
   CheckTakeProfit();

   if(!UseDQN)
      return;

   tickCounter++;
   if(isTraining && TrainingFreq>0 && (tickCounter % TrainingFreq)==0)
   {
      if(currentEpsilon > MinExplorationRate)
         currentEpsilon *= ExplorationDecay;

      if(currentEpsilon < MinExplorationRate)
         currentEpsilon = MinExplorationRate;
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      ManagePairWithDQN(gSymbols[i], i, gPositionsCount[i], gTrades[i], gMagics[i], gFirstTradeTime[i]);
      if(!MQLInfoInteger(MQL_TESTER)) UpdateGridStatusPanel(i);
   }
}

//+------------------------------------------------------------------+
void InitOrRandomizeDQN(int symIdx,int regime,int inputDim)
{
   if(inputDim <= 0) inputDim = 8;

   if(UseBranchScaffold)
      InitBranchLayoutForStateDim(inputDim);

   DQNNetwork net = gDQN[symIdx][regime];

   net.input_dim  = inputDim;
   net.hidden_dim = (HiddenSize>0 ? HiddenSize : 24);
   net.hidden_dim2= (HiddenSize2>0 ? HiddenSize2 : 0);
   net.output_dim = (ActionCount>0 ? ActionCount : 3);

   net.basket_h1 = (gBranchLayout.basketCount>0 ? BranchHidden1Size(gBranchLayout.basketCount) : 0);
   net.basket_h2 = (gBranchLayout.basketCount>0 ? BranchHidden2Size(gBranchLayout.basketCount, net.basket_h1) : 0);

   net.indicator_h1 = (gBranchLayout.indicatorCount>0 ? BranchHidden1Size(gBranchLayout.indicatorCount) : 0);
   net.indicator_h2 = (gBranchLayout.indicatorCount>0 ? BranchHidden2Size(gBranchLayout.indicatorCount, net.indicator_h1) : 0);

   net.volatility_h1 = (gBranchLayout.volatilityCount>0 ? BranchHidden1Size(gBranchLayout.volatilityCount) : 0);
   net.volatility_h2 = (gBranchLayout.volatilityCount>0 ? BranchHidden2Size(gBranchLayout.volatilityCount, net.volatility_h1) : 0);

   net.structure_h1 = (gBranchLayout.structureCount>0 ? BranchHidden1Size(gBranchLayout.structureCount) : 0);
   net.structure_h2 = (gBranchLayout.structureCount>0 ? BranchHidden2Size(gBranchLayout.structureCount, net.structure_h1) : 0);

   net.zone_h1 = (gBranchLayout.zoneCandleCount>0 ? BranchHidden1Size(gBranchLayout.zoneCandleCount) : 0);
   net.zone_h2 = (gBranchLayout.zoneCandleCount>0 ? BranchHidden2Size(gBranchLayout.zoneCandleCount, net.zone_h1) : 0);

   net.fusion_dim = net.basket_h2 + net.indicator_h2 + net.volatility_h2 + net.structure_h2 + net.zone_h2;
   if(net.fusion_dim <= 0) net.fusion_dim = net.input_dim;

   InitBranchEncoder(gBranchLayout.basketCount, net.basket_h1, net.basket_h2, net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2);
   InitBranchEncoder(gBranchLayout.indicatorCount, net.indicator_h1, net.indicator_h2, net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2);
   InitBranchEncoder(gBranchLayout.volatilityCount, net.volatility_h1, net.volatility_h2, net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2);
   InitBranchEncoder(gBranchLayout.structureCount, net.structure_h1, net.structure_h2, net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2);
   InitBranchEncoder(gBranchLayout.zoneCandleCount, net.zone_h1, net.zone_h2, net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2);

   int fusionDim = net.fusion_dim;
   int hidDim1 = net.hidden_dim;
   int hidDim2 = net.hidden_dim2;
   int headDim = (hidDim2>0 ? hidDim2 : hidDim1);
   int outDim = net.output_dim;

   ArrayResize(net.W1, hidDim1*fusionDim);
   ArrayResize(net.b1, hidDim1);
   ArrayResize(net.W2, MathMax(0,hidDim2*hidDim1));
   ArrayResize(net.b2, MathMax(0,hidDim2));

   ArrayResize(net.WV, headDim);
   ArrayResize(net.bV, 1);

   ArrayResize(net.WA, outDim*headDim);
   ArrayResize(net.bA, outDim);

   ArrayResize(net.feat_mean, net.input_dim);
   ArrayResize(net.feat_std,  net.input_dim);

   double scale1 = 1.0 / MathSqrt((double)MathMax(fusionDim,1));
   double scale2 = 1.0 / MathSqrt((double)MathMax(hidDim1,1));
   double scaleV = 1.0 / MathSqrt((double)MathMax(headDim,1));
   double scaleA = 1.0 / MathSqrt((double)MathMax(headDim,1));

   for(int i=0;i<ArraySize(net.W1);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.W1[i] = (r*2.0-1.0)*scale1;
   }
   for(int i=0;i<ArraySize(net.b1);i++) net.b1[i]=0.0;

   for(int i=0;i<ArraySize(net.W2);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.W2[i] = (r*2.0-1.0)*scale2;
   }
   for(int i=0;i<ArraySize(net.b2);i++) net.b2[i]=0.0;

   for(int i=0;i<ArraySize(net.WV);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.WV[i] = (r*2.0-1.0)*scaleV;
   }
   for(int i=0;i<ArraySize(net.bV);i++) net.bV[i]=0.0;

   for(int i=0;i<ArraySize(net.WA);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.WA[i] = (r*2.0-1.0)*scaleA;
   }
   for(int i=0;i<ArraySize(net.bA);i++) net.bA[i]=0.0;

   for(int i=0;i<net.input_dim;i++)
   {
      net.feat_mean[i]=0.0;
      net.feat_std[i]=1.0;
   }

   gDQN[symIdx][regime] = net;
   SyncTargetNetFor(symIdx, regime);
}
