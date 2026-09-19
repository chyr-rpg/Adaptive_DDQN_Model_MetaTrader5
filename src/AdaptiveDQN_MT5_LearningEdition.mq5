//+------------------------------------------------------------------+
//| Adaptive DQN for MetaTrader 5 - Learning Edition                 |
//|                                                                  |
//| Public educational implementation of the foundational neural     |
//| reinforcement-learning architecture used in Adaptive-DDQN-MT5.   |
//|                                                                  |
//| Includes:                                                        |
//|  - Native MQL5 DQN with two hidden layers                        |
//|  - HOLD / BUY / SELL action space                                |
//|  - Epsilon-greedy learning                                       |
//|  - Multi-asset state construction                                |
//|  - Volatility / HTF / supply-demand features                     |
//|  - Risk-aware reward shaping                                     |
//|  - Persistent neural-network state                               |
//|  - Visual Strategy Tester learning monitor                       |
//|                                                                  |
//| The current full research implementation is maintained privately.|
//| See the public repository documentation for architecture details.|
//+------------------------------------------------------------------+
#property copyright "2025-2026, Chen Yurui"
#property version   "1.10"
#property strict

#include <Trade/Trade.mqh>
#include <Arrays/ArrayDouble.mqh>

CTrade trade;

//--- Max symbols
#define MAX_SYMBOLS 16

//--- Symbol state
int          gSymbolCount                 = 0;
string       gSymbols[MAX_SYMBOLS];
int          gMagics[MAX_SYMBOLS];
int          gPositionsCount[MAX_SYMBOLS];
CArrayDouble gTrades[MAX_SYMBOLS];
datetime     gFirstTradeTime[MAX_SYMBOLS];

//==================================================================
//  SUPPLY/DEMAND ZONES
//==================================================================
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

//==================================================================
//  INPUTS
//==================================================================

//--- Symbols / Universe
enum ENUM_SYMBOL_MODE { SYM_CURRENT=0, SYM_LIST=1, SYM_MARKETWATCH=2 };

input string           SymbolsSettings = "==== Symbols / Universe ====";
input ENUM_SYMBOL_MODE SymbolMode      = SYM_CURRENT;
input string           SymbolsList     = "EURUSD,GBPUSD,XAUUSD,US30";
input int              BaseMagic       = 220111;

//--- Logic timeframe
input string          TFSettings       = "==== Timeframe settings ====";
input ENUM_TIMEFRAMES BaseTF           = PERIOD_M5;

//--- General
input string   General              = "==== General settings ====";
input double   Lots                 = 0.01;
input double   LotExponent          = 1.4;
input int      MaxTrades            = 10;
input int      TakeProfit           = 100;     // POINTS
input int      StopLoss             = 0;       // POINTS
input int      Slippage             = 30;

//--- Q/DQN base settings
input string   SymbolSettings       = "==== Q/DQN settings ====";
input string   QTablePrefix         = "";

//--- Dynamic grid
input string   DynamicSettings      = "==== Dynamic grid / channel settings ====";
input bool     UseDynamicPips       = true;
input int      DefaultPips          = 120;     // POINTS
input int      Depth                = 24;
input double   PipsFactor           = 3.0;

//--- Indicators
input string   IndicatorSettings    = "==== Indicator settings ====";
input int      RSI_Period           = 14;
input double   RSI_Minimum          = 30.0;
input double   RSI_Maximum          = 70.0;
input bool     UseCCI               = false;
input int      CCI_Period           = 55;
input int      CCI_Level            = 500;

//--- Exit / risk
input string   ExitSettings         = "==== Exit & risk settings ====";
input bool     UseTrailingStop      = false;
input int      TrailStart           = 100;
input int      TrailStop            = 100;
input bool     UseEquityStop        = false;
input double   EquityRiskPercent    = 20.0;

//--- Virtual equity budget
input string   BudgetSettings       = "==== Virtual equity budget ====";
input double   EquityBudget         = 1000.0;

//--- Correlation filter (optional; only meaningful if >=2 symbols)
input string   CorrelationSettings  = "==== Correlation filter settings ====";
input bool     UseCorrelation       = true;
input double   MinCorrelation       = 0.7;
input int      CorrelationPeriod    = 50;

//--- RL settings
input string   DQNSettings          = "==== RL control settings ====";
input bool     UseDQN               = true;
input int      StateDimension       = 10;     // features used
input int      ActionCount          = 3;      // 0=hold,1=buy,2=sell
input double   ExplorationRate      = 0.3;
input double   ExplorationDecay     = 0.995;
input double   MinExplorationRate   = 0.01;
input int      TrainingFreq         = 10;
input bool     SaveQTable           = true;

//--- DQN network settings (TWO hidden layers)
input string   DQNNetSettings       = "==== Real DQN network settings ====";
input int      HiddenSize           = 16;
input int      HiddenSize2          = 16;
input double   DQNLearningRate      = 0.001;
input double   DQNGamma             = 0.95;
input bool     UseInputNorm         = false;

//--- Volatility regime (ATR)
input string   RegimeSettings       = "==== Volatility / ATR settings ====";
input int      ATR_FastPeriod       = 14;
input int      ATR_SlowPeriod       = 100;

//--- RL reward shaping
input string   RewardSettings       = "==== RL reward shaping settings ====";
input double   EquityRewardScale    = 50.0;
input double   DrawdownPenaltyScale = 10.0;
input double   TrendPenaltyThreshold= 1.0;
input double   TrendPenaltyFactor   = 5.0;

//--- RL regime weighting
input string   RegimeWeightSettings = "==== RL regime weighting ====";
input double   ExtremeRewardBoost   = 3.0;
input double   MildRewardScale      = 0.5;
input bool     SeparateExtremeStates= true;

//--- Training mode control
input string   TrainingSettings     = "==== Training mode ====";
input bool     TrainingMode         = true;

//--- Zone settings
input string   ZoneSettings         = "==== Retest Supply/Demand zone settings ====";
input int      MinZoneBaseBars      = 3;
input int      MaxZoneBaseBars      = 8;
input double   ZoneMaxHeightPoints  = 300;
input int      ZoneExtendBars       = 200;

//--- D1 EMA trend
input string   D1TrendSettings      = "==== D1 EMA trend settings ====";
input int      D1_EMA_Period        = 100;
input int      D1_SlopeLookbackDays = 20;
input int      D1_SideLookbackDays  = 60;
input double   D1_StrongTrendDistance = 1.0;
input double   D1_MaxSideDurationDays = 60.0;
input double   D1_MaxGridWidenFactor  = 3.0;

//--- HTF trend (CCI/ATR)
input string          HTFTrendSettings   = "==== HTF trend settings (CCI/ATR) ====";
input ENUM_TIMEFRAMES HTF_Timeframe      = PERIOD_H4;
input int             HTF_CCI_Period     = 55;
input int             HTF_ATR_Period     = 14;
input double          HTF_ATR_ScalePoints= 2000.0;

//--- Performance / viz
input string PerformanceSettings  = "==== Performance / visualization ====";
input bool   DrawZonesOnChart     = true;
input bool   VerboseLogging       = false;

//--- Learning dashboard
input bool   ShowLearningDashboard = true;
input string DashboardSymbol       = "";   // blank = chart symbol / first available

//==================================================================
//  INTERNAL EA VIRTUAL EQUITY STATE
//==================================================================
double   maxEquity            = 0.0;
double   gLastEquityForReward = 0.0;
double   gEAStartEquity       = 0.0;
double   gEAClosedProfit      = 0.0;

//--- RL runtime
int      tickCounter          = 0;
double   currentEpsilon       = 0.0;
bool     isTraining           = true;
double   totalReward          = 0.0;
int      episodeCount         = 0;

//==================================================================
//  LEARNING DASHBOARD STATE
//==================================================================
int    gLastSelectedAction[MAX_SYMBOLS];
bool   gLastActionExploratory[MAX_SYMBOLS];
bool   gHasActionDecision[MAX_SYMBOLS];
double gLastQHold[MAX_SYMBOLS];
double gLastQBuy[MAX_SYMBOLS];
double gLastQSell[MAX_SYMBOLS];

//==================================================================
//  REAL DQN NETWORK STRUCTURE (TWO HIDDEN LAYERS)
//==================================================================
struct DQNNetwork
{
   int    input_dim;
   int    hidden1_dim;
   int    hidden2_dim;
   int    output_dim;

   double W1[];   // hidden1_dim * input_dim
   double b1[];   // hidden1_dim
   double W2[];   // hidden2_dim * hidden1_dim
   double b2[];   // hidden2_dim
   double W3[];   // output_dim * hidden2_dim
   double b3[];   // output_dim
   double feat_mean[];
   double feat_std[];
};

DQNNetwork gDQN[MAX_SYMBOLS];

int W1Index(DQNNetwork &net, int h1, int i)   { return h1*net.input_dim    + i;  }
int W2Index(DQNNetwork &net, int h2, int h1)  { return h2*net.hidden1_dim  + h1; }
int W3Index(DQNNetwork &net, int o,  int h2)  { return o *net.hidden2_dim  + h2; }

//==================================================================
//  PERFORMANCE CACHES
//==================================================================
datetime gLastBarTime[MAX_SYMBOLS];

datetime gD1LastBarTime[MAX_SYMBOLS];
double   gD1DistCache      [MAX_SYMBOLS];
double   gD1SlopeCache     [MAX_SYMBOLS];
double   gD1SideDurCache   [MAX_SYMBOLS];

datetime gHTFLastBarTime[MAX_SYMBOLS];
double   gHTFCCICache[MAX_SYMBOLS];
double   gHTFATRCache[MAX_SYMBOLS];

datetime gATRLastBarTime[MAX_SYMBOLS];
double   gATRratioCache[MAX_SYMBOLS];

double   gLastCorrelation = 1.0;
datetime gLastCorrBarTime = 0;

//+------------------------------------------------------------------+
//| Helpers                                                         |
//+------------------------------------------------------------------+
string Trim(string s){ StringTrimLeft(s); StringTrimRight(s); return s; }

int SplitCSV(const string csv, string &out[])
{
   ArrayResize(out,0);
   int n=StringSplit(csv, ',', out);
   for(int i=0;i<n;i++) out[i]=Trim(out[i]);
   return n;
}

bool IsTradableSymbol(const string sym)
{
   if(sym=="" ) return false;
   if(!SymbolInfoInteger(sym, SYMBOL_SELECT)) SymbolSelect(sym,true);

   long tradeMode = SymbolInfoInteger(sym, SYMBOL_TRADE_MODE);
   if(tradeMode == SYMBOL_TRADE_MODE_DISABLED) return false;

   double bid=SymbolInfoDouble(sym,SYMBOL_BID);
   double ask=SymbolInfoDouble(sym,SYMBOL_ASK);
   if(bid<=0 || ask<=0) return false;

   return true;
}

int MagicForSymbol(const string sym)
{
   uint h=2166136261;
   for(int i=0;i<StringLen(sym);i++)
      h = (h ^ (uchar)StringGetCharacter(sym,i)) * 16777619;
   return BaseMagic + (int)(h % 100000);
}

void BuildSymbolUniverse()
{
   gSymbolCount=0;

   if(SymbolMode==SYM_CURRENT)
   {
      if(IsTradableSymbol(_Symbol))
      {
         gSymbols[0]=_Symbol;
         gMagics[0]=MagicForSymbol(_Symbol);
         gSymbolCount=1;
      }
      return;
   }

   if(SymbolMode==SYM_LIST)
   {
      string list[];
      int n=SplitCSV(SymbolsList, list);
      for(int i=0;i<n && gSymbolCount<MAX_SYMBOLS;i++)
      {
         string sym=list[i];
         if(!IsTradableSymbol(sym)) continue;
         gSymbols[gSymbolCount]=sym;
         gMagics[gSymbolCount]=MagicForSymbol(sym);
         gSymbolCount++;
      }
      return;
   }

   // SYM_MARKETWATCH
   int total=SymbolsTotal(true);
   for(int i=0;i<total && gSymbolCount<MAX_SYMBOLS;i++)
   {
      string sym=SymbolName(i,true);
      if(!IsTradableSymbol(sym)) continue;
      gSymbols[gSymbolCount]=sym;
      gMagics[gSymbolCount]=MagicForSymbol(sym);
      gSymbolCount++;
   }
}

int SymbolIndex(const string symbol)
{
   for(int i=0;i<gSymbolCount;i++)
      if(gSymbols[i]==symbol) return i;
   return -1;
}

double Clamp(const double v,const double lo,const double hi)
{
   if(v<lo) return lo;
   if(v>hi) return hi;
   return v;
}

double SafeDiv(const double a,const double b,const double fallback=0.0)
{
   if(MathAbs(b)<=1e-12) return fallback;
   return a/b;
}

double NormalizeVolume(const string symbol, double vol)
{
   double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vmax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(vstep<=0.0) vstep=0.01;
   if(vmin<=0.0)  vmin=vstep;
   if(vmax<=0.0)  vmax=vol;

   vol = MathMax(vmin, MathMin(vmax, vol));
   double steps = MathFloor(vol / vstep + 1e-9);
   double out   = steps * vstep;

   int digits=0;
   if(vstep<1.0)
   {
      double lg=-MathLog10(vstep);
      if(lg<0) lg=0;
      digits=(int)MathCeil(lg);
      if(digits>8) digits=8;
   }
   return NormalizeDouble(out, digits);
}

// --- cheap indicator value getter (create handle, copy 1 value, release)
bool Copy1(const int handle, const int buffer, double &outVal)
{
   if(handle==INVALID_HANDLE) return false;
   double tmp[1];
   if(CopyBuffer(handle, buffer, 0, 1, tmp) <= 0) return false;
   outVal=tmp[0];
   return true;
}

double GetRSIValue(const string symbol, ENUM_TIMEFRAMES tf, int period, int shift=0)
{
   int h=iRSI(symbol, tf, period, PRICE_CLOSE);
   if(h==INVALID_HANDLE) return 50.0;
   double buf[];
   ArrayResize(buf, shift+1);
   ArraySetAsSeries(buf,true);
   double v=50.0;
   if(CopyBuffer(h,0,0,shift+1,buf)>0) v=buf[shift];
   IndicatorRelease(h);
   return v;
}

double GetCCIValue(const string symbol, ENUM_TIMEFRAMES tf, int period, int shift=0)
{
   int h=iCCI(symbol, tf, period, PRICE_TYPICAL);
   if(h==INVALID_HANDLE) return 0.0;
   double buf[];
   ArrayResize(buf, shift+1);
   ArraySetAsSeries(buf,true);
   double v=0.0;
   if(CopyBuffer(h,0,0,shift+1,buf)>0) v=buf[shift];
   IndicatorRelease(h);
   return v;
}

double GetMACDMain(const string symbol, ENUM_TIMEFRAMES tf, int fast=12, int slow=26, int sig=9, int shift=0)
{
   int h=iMACD(symbol, tf, fast, slow, sig, PRICE_CLOSE);
   if(h==INVALID_HANDLE) return 0.0;
   double buf[];
   ArrayResize(buf, shift+1);
   ArraySetAsSeries(buf,true);
   double v=0.0;
   if(CopyBuffer(h,0,0,shift+1,buf)>0) v=buf[shift]; // main line buffer=0
   IndicatorRelease(h);
   return v;
}

bool CheckStopsDistanceOk(const string symbol, double price, double sl, double tp)
{
   long stops = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point<=0.0) return true;
   double minDist = (double)stops * point;

   if(sl>0.0 && MathAbs(price-sl) < minDist) return false;
   if(tp>0.0 && MathAbs(price-tp) < minDist) return false;
   return true;
}

//==================================================================
//  ZONE FEATURE BUILDER
//==================================================================
void GetZoneFeatures(string symbol,double lastClose,
                     double &zoneTypeNorm,double &zoneLocNorm,double &zoneStatusNorm)
{
   zoneTypeNorm   = 0.5;
   zoneLocNorm    = 0.5;
   zoneStatusNorm = 0.5;

   int total = ArraySize(zones);
   if(total <= 0) return;

   int bestIdx=-1;
   double bestDist=DBL_MAX;
   datetime now=TimeCurrent();

   for(int i=0;i<total;i++)
   {
      if(zones[i].symbol!=symbol) continue;
      if(now>zones[i].endTime) continue;

      double centre=0.5*(zones[i].high+zones[i].low);
      double dist=MathAbs(lastClose-centre);
      if(dist<bestDist){ bestDist=dist; bestIdx=i; }
   }
   if(bestIdx<0) return;

   SDZone z=zones[bestIdx];
   double h=z.high, l=z.low;
   double height=h-l;
   if(height<=0.0) return;

   zoneTypeNorm = z.isDemand ? 0.0 : 1.0;

   double loc=(lastClose-l)/height;
   zoneLocNorm = Clamp(loc,0.0,1.0);

   if(z.broken)      zoneStatusNorm=1.0;
   else if(z.tested) zoneStatusNorm=0.5;
   else              zoneStatusNorm=0.0;
}

//==================================================================
//  D1 TREND FEATURES
//==================================================================
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
   if(idx<0){ d1DistNorm=0; d1SlopeNorm=0; d1SideDurNorm=0; return; }

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

//==================================================================
//  HTF TREND FEATURES
//==================================================================
void ComputeHTFTrendFeaturesRaw(string symbol, ENUM_TIMEFRAMES tf, int cciPeriod, int atrPeriod,
                                double &htfCCINorm, double &htfATRNorm)
{
   htfCCINorm=0.5;
   htfATRNorm=0.0;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.0001;

   if(cciPeriod>0)
   {
      double cciVal=GetCCIValue(symbol, tf, cciPeriod, 0);
      cciVal=Clamp(cciVal,-500.0,500.0);
      htfCCINorm=(cciVal/1000.0)+0.5;
   }

   if(atrPeriod>0)
   {
      int h=iATR(symbol, tf, atrPeriod);
      if(h!=INVALID_HANDLE)
      {
         double buf[1];
         if(CopyBuffer(h,0,0,1,buf)>0)
         {
            double atrVal=buf[0];
            double scale=HTF_ATR_ScalePoints*point;
            if(scale<=0.0) scale=point*1000.0;
            double norm=Clamp(atrVal/scale,0.0,5.0);
            htfATRNorm=norm/5.0;
         }
         IndicatorRelease(h);
      }
   }
}

void GetHTFTrendFeatures(string symbol, ENUM_TIMEFRAMES tf, int cciPeriod, int atrPeriod,
                         double &htfCCINorm, double &htfATRNorm)
{
   int idx=SymbolIndex(symbol);
   if(idx<0){ htfCCINorm=0.5; htfATRNorm=0.0; return; }

   datetime bt=iTime(symbol, tf, 0);
   if(bt!=gHTFLastBarTime[idx])
   {
      ComputeHTFTrendFeaturesRaw(symbol, tf, cciPeriod, atrPeriod, gHTFCCICache[idx], gHTFATRCache[idx]);
      gHTFLastBarTime[idx]=bt;
   }
   htfCCINorm=gHTFCCICache[idx];
   htfATRNorm=gHTFATRCache[idx];
}

//==================================================================
//  VIRTUAL EQUITY HELPERS
//==================================================================
double GetEAOpenPnL()
{
   double totalPnL=0.0;
   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym=PositionGetString(POSITION_SYMBOL);
      long mg=PositionGetInteger(POSITION_MAGIC);

      int idx=SymbolIndex(sym);
      if(idx<0) continue;
      if((int)mg!=gMagics[idx]) continue;

      totalPnL += PositionGetDouble(POSITION_PROFIT);
   }
   return totalPnL;
}

double GetEAEquity()
{
   return gEAStartEquity + gEAClosedProfit + GetEAOpenPnL();
}

//==================================================================
//  STATE BUILDER (numeric only)
//==================================================================
void GetCurrentState(string symbol,int positionsCount,CArrayDouble &trades,double &state[])
{
   const int FEATURES=17;
   int dim=(StateDimension<FEATURES ? StateDimension : FEATURES);
   if(dim<=0){ ArrayResize(state,0); return; }
   ArrayResize(state, dim);

   int bars=iBars(symbol, BaseTF);
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double rsi = GetRSIValue(symbol, BaseTF, RSI_Period, 0);
   double cci = GetCCIValue(symbol, BaseTF, CCI_Period, 0);
   double macd= GetMACDMain(symbol, BaseTF, 12,26,9, 0);

   double normalized_rsi = Clamp(rsi/100.0,0.0,1.0);
   double normalized_cci = Clamp((cci+500.0)/1000.0,0.0,1.0);
   double normalized_positions = (MaxTrades>0 ? Clamp((double)positionsCount/(double)MaxTrades,0.0,1.0) : 0.0);

   double avgPrice=bid;
   if(positionsCount>0 && trades.Total()>0)
   {
      avgPrice=0.0;
      for(int i=0;i<trades.Total();i++) avgPrice += trades.At(i);
      avgPrice /= (double)trades.Total();
   }
   double price_diff = SafeDiv((bid-avgPrice),(100.0*point),0.0);

   // ATR ratio cached on BaseTF
   double atr_ratio=1.0;
   int idxATR=SymbolIndex(symbol);
   if(idxATR>=0)
   {
      datetime bt=iTime(symbol, BaseTF, 0);
      if(bt!=gATRLastBarTime[idxATR])
      {
         double tmp=1.0;
         int hF=iATR(symbol, BaseTF, ATR_FastPeriod);
         int hS=iATR(symbol, BaseTF, ATR_SlowPeriod);
         if(hF!=INVALID_HANDLE && hS!=INVALID_HANDLE)
         {
            double f[1], s[1];
            if(CopyBuffer(hF,0,0,1,f)>0 && CopyBuffer(hS,0,0,1,s)>0 && s[0]>0.0)
               tmp=Clamp(f[0]/s[0],0.1,5.0);
            IndicatorRelease(hF);
            IndicatorRelease(hS);
         }
         gATRratioCache[idxATR]=tmp;
         gATRLastBarTime[idxATR]=bt;
      }
      atr_ratio=gATRratioCache[idxATR];
   }

   double eq=GetEAEquity();
   double bal=gEAStartEquity;
   double eq_bal = (bal>1e-9 ? eq/bal : 1.0);

   double dd_ratio=0.0;
   if(maxEquity>1e-9 && eq<maxEquity)
      dd_ratio=Clamp((maxEquity-eq)/maxEquity,0.0,1.0);

   double lastClose=iClose(symbol, BaseTF, 1);
   double zoneTypeNorm=0.5, zoneLocNorm=0.5, zoneStatusNorm=0.5;
   GetZoneFeatures(symbol,lastClose,zoneTypeNorm,zoneLocNorm,zoneStatusNorm);

   // short/long return in BaseTF
   double short_ret=0.0;
   int shortLB=30;
   if(shortLB>1 && bars>shortLB)
   {
      double c0=iClose(symbol, BaseTF, 1);
      double c1=iClose(symbol, BaseTF, shortLB);
      short_ret=Clamp((c0-c1)/(point*shortLB),-5.0,5.0);
   }

   double long_ret=0.0;
   int longLB=240;
   if(longLB>1 && bars>longLB)
   {
      double c0=iClose(symbol, BaseTF, 1);
      double c1=iClose(symbol, BaseTF, longLB);
      long_ret=Clamp((c0-c1)/(point*longLB),-5.0,5.0);
   }

   int downStreak=0;
   int maxStreak=60;
   if(bars>2)
   {
      int check=MathMin(maxStreak, bars-2);
      for(int k=1;k<=check;k++)
      {
         double c_now=iClose(symbol, BaseTF, k);
         double c_prev=iClose(symbol, BaseTF, k+1);
         if(c_now<c_prev) downStreak++;
         else break;
      }
   }
   double down_streak_norm=Clamp((double)downStreak/(double)maxStreak,0.0,1.0);

   double d1DistNorm=0.0,d1SlopeNorm=0.0,d1SideDurNorm=0.0;
   GetD1TrendFeatures(symbol,d1DistNorm,d1SlopeNorm,d1SideDurNorm);

   double htfCCINorm=0.5, htfATRNorm=0.0;
   GetHTFTrendFeatures(symbol, HTF_Timeframe, HTF_CCI_Period, HTF_ATR_Period, htfCCINorm, htfATRNorm);

   if(dim>0)  state[0]=normalized_rsi;
   if(dim>1)  state[1]=normalized_cci;
   if(dim>2)  state[2]=normalized_positions;
   if(dim>3)  state[3]=price_diff;
   if(dim>4)  state[4]=SafeDiv(macd,(100.0*point),0.0);
   if(dim>5)  state[5]=eq_bal;
   if(dim>6)  state[6]=atr_ratio;
   if(dim>7)  state[7]=zoneStatusNorm;
   if(dim>8)  state[8]=short_ret;
   if(dim>9)  state[9]=long_ret;
   if(dim>10) state[10]=down_streak_norm;
   if(dim>11) state[11]=d1DistNorm;
   if(dim>12) state[12]=d1SlopeNorm;
   if(dim>13) state[13]=d1SideDurNorm;
   if(dim>14) state[14]=dd_ratio;
   if(dim>15) state[15]=htfCCINorm;
   if(dim>16) state[16]=htfATRNorm;

   if(SeparateExtremeStates)
   {
      if(dim<FEATURES)
      {
         int newDim=dim+1;
         ArrayResize(state,newDim);
         dim=newDim;
      }
      int sz=ArraySize(state);
      if(sz>0) state[sz-1]=0.0;
   }
}

//==================================================================
//  EXTREME STATE + REWARD SCALING
//==================================================================
bool IsExtremeState(double &state[])
{
   int sz=ArraySize(state);
   if(sz==0) return false;

   bool extreme=false;
   if(sz>6 && state[6]>2.0) extreme=true;
   if(sz>8 && MathAbs(state[8])>2.0) extreme=true;
   if(sz>9 && MathAbs(state[9])>2.0) extreme=true;
   if(sz>11 && MathAbs(state[11])>2.0) extreme=true;
   return extreme;
}

double ScaleRewardByRegime(double reward, bool isExtreme)
{
   if(isExtreme) return reward * ExtremeRewardBoost;
   if(reward < 0.0) return reward * MildRewardScale;
   return reward;
}

//==================================================================
//  REAL DQN: FORWARD, UPDATE, ACTION SELECTION, SAVE/LOAD
//==================================================================
void DQNForward(int symIdx, double &state[], double &qOut[])
{
   int inDim=gDQN[symIdx].input_dim;
   int h1Dim=gDQN[symIdx].hidden1_dim;
   int h2Dim=gDQN[symIdx].hidden2_dim;
   int outDim=gDQN[symIdx].output_dim;

   ArrayResize(qOut,outDim);

   double x[]; ArrayResize(x,inDim);
   for(int i=0;i<inDim;i++)
   {
      double v=(i<ArraySize(state)?state[i]:0.0);
      if(UseInputNorm && i<ArraySize(gDQN[symIdx].feat_std) && gDQN[symIdx].feat_std[i]>0.0)
         v=(v-gDQN[symIdx].feat_mean[i])/gDQN[symIdx].feat_std[i];
      x[i]=v;
   }

   double h1[]; ArrayResize(h1,h1Dim);
   for(int h=0;h<h1Dim;h++)
   {
      double sum=gDQN[symIdx].b1[h];
      for(int i=0;i<inDim;i++)
         sum += gDQN[symIdx].W1[W1Index(gDQN[symIdx],h,i)] * x[i];
      h1[h]=MathTanh(sum);
   }

   double h2[]; ArrayResize(h2,h2Dim);
   for(int j=0;j<h2Dim;j++)
   {
      double sum=gDQN[symIdx].b2[j];
      for(int h=0;h<h1Dim;h++)
         sum += gDQN[symIdx].W2[W2Index(gDQN[symIdx],j,h)] * h1[h];
      h2[j]=MathTanh(sum);
   }

   for(int o=0;o<outDim;o++)
   {
      double sum=gDQN[symIdx].b3[o];
      for(int j=0;j<h2Dim;j++)
         sum += gDQN[symIdx].W3[W3Index(gDQN[symIdx],o,j)] * h2[j];
      qOut[o]=sum;
   }
}

void DQNUpdate(int symIdx, double &state[], int action, double reward, double &nextState[], bool done)
{
   int inDim=gDQN[symIdx].input_dim;
   int h1Dim=gDQN[symIdx].hidden1_dim;
   int h2Dim=gDQN[symIdx].hidden2_dim;
   int outDim=gDQN[symIdx].output_dim;

   double lr=DQNLearningRate;

   double x[]; ArrayResize(x,inDim);
   for(int i=0;i<inDim;i++)
   {
      double v=(i<ArraySize(state)?state[i]:0.0);
      if(UseInputNorm && i<ArraySize(gDQN[symIdx].feat_std) && gDQN[symIdx].feat_std[i]>0.0)
         v=(v-gDQN[symIdx].feat_mean[i])/gDQN[symIdx].feat_std[i];
      x[i]=v;
   }

   double h1[]; ArrayResize(h1,h1Dim);
   for(int h=0;h<h1Dim;h++)
   {
      double sum=gDQN[symIdx].b1[h];
      for(int i=0;i<inDim;i++)
         sum += gDQN[symIdx].W1[W1Index(gDQN[symIdx],h,i)] * x[i];
      h1[h]=MathTanh(sum);
   }

   double h2[]; ArrayResize(h2,h2Dim);
   for(int j=0;j<h2Dim;j++)
   {
      double sum=gDQN[symIdx].b2[j];
      for(int h=0;h<h1Dim;h++)
         sum += gDQN[symIdx].W2[W2Index(gDQN[symIdx],j,h)] * h1[h];
      h2[j]=MathTanh(sum);
   }

   double q[]; ArrayResize(q,outDim);
   for(int o=0;o<outDim;o++)
   {
      double sum=gDQN[symIdx].b3[o];
      for(int j=0;j<h2Dim;j++)
         sum += gDQN[symIdx].W3[W3Index(gDQN[symIdx],o,j)] * h2[j];
      q[o]=sum;
   }

   double target=reward;
   if(!done)
   {
      double qNext[];
      DQNForward(symIdx,nextState,qNext);
      double maxNext=qNext[0];
      for(int o=1;o<outDim;o++) if(qNext[o]>maxNext) maxNext=qNext[o];
      target += DQNGamma * maxNext;
   }

   double tdError=target - q[action];
   double dQ=-tdError;

   // output layer update (chosen action)
   for(int j=0;j<h2Dim;j++)
   {
      int idxW3=W3Index(gDQN[symIdx],action,j);
      gDQN[symIdx].W3[idxW3] -= lr * (dQ * h2[j]);
   }
   gDQN[symIdx].b3[action] -= lr * dQ;

   // backprop hidden2
   double dH2[]; ArrayResize(dH2,h2Dim);
   for(int j=0;j<h2Dim;j++)
   {
      double dSum = dQ * gDQN[symIdx].W3[W3Index(gDQN[symIdx],action,j)];
      dH2[j] = dSum * (1.0 - h2[j]*h2[j]);
   }

   // update W2,b2
   for(int j=0;j<h2Dim;j++)
   {
      for(int h=0;h<h1Dim;h++)
      {
         int idxW2=W2Index(gDQN[symIdx],j,h);
         gDQN[symIdx].W2[idxW2] -= lr * (dH2[j] * h1[h]);
      }
      gDQN[symIdx].b2[j] -= lr * dH2[j];
   }

   // backprop hidden1
   double dH1[]; ArrayResize(dH1,h1Dim);
   for(int h=0;h<h1Dim;h++)
   {
      double sum=0.0;
      for(int j=0;j<h2Dim;j++)
         sum += dH2[j] * gDQN[symIdx].W2[W2Index(gDQN[symIdx],j,h)];
      dH1[h] = sum * (1.0 - h1[h]*h1[h]);
   }

   // update W1,b1
   for(int h=0;h<h1Dim;h++)
   {
      for(int i=0;i<inDim;i++)
      {
         int idxW1=W1Index(gDQN[symIdx],h,i);
         gDQN[symIdx].W1[idxW1] -= lr * (dH1[h] * x[i]);
      }
      gDQN[symIdx].b1[h] -= lr * dH1[h];
   }
}

int DQNSelectAction(int symIdx, double &state[])
{
   bool exploring =
      ((double)MathRand()/32767.0 < currentEpsilon);

   // Always snapshot the policy Q-values that existed at decision time.
   // This keeps the visual dashboard aligned with the action that was
   // actually selected, even if a training update occurs later in the tick.
   double q[];
   DQNForward(symIdx,state,q);

   gLastQHold[symIdx]=(ArraySize(q)>0 ? q[0] : 0.0);
   gLastQBuy[symIdx] =(ArraySize(q)>1 ? q[1] : 0.0);
   gLastQSell[symIdx]=(ArraySize(q)>2 ? q[2] : 0.0);

   int action=0;

   if(exploring)
   {
      action = MathRand() % ActionCount;
      gLastActionExploratory[symIdx] = true;
   }
   else
   {
      int best=0;
      double maxQ=q[0];

      for(int a=1;a<ActionCount;a++)
      {
         if(q[a]>maxQ)
         {
            maxQ=q[a];
            best=a;
         }
      }

      action=best;
      gLastActionExploratory[symIdx] = false;
   }

   gLastSelectedAction[symIdx] = action;
   gHasActionDecision[symIdx] = true;

   return action;
}

bool SaveDQNForSymbol(int symIdx, string filename)
{
   int handle=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(handle==INVALID_HANDLE)
   {
      Print("Error opening DQN file for writing: ",filename," err=",GetLastError());
      return false;
   }

   int version=2;
   FileWriteInteger(handle,version);
   FileWriteInteger(handle,gDQN[symIdx].input_dim);
   FileWriteInteger(handle,gDQN[symIdx].hidden1_dim);
   FileWriteInteger(handle,gDQN[symIdx].hidden2_dim);
   FileWriteInteger(handle,gDQN[symIdx].output_dim);

   int sz;
   sz=ArraySize(gDQN[symIdx].W1); FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].W1[i]);
   sz=ArraySize(gDQN[symIdx].b1); FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].b1[i]);
   sz=ArraySize(gDQN[symIdx].W2); FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].W2[i]);
   sz=ArraySize(gDQN[symIdx].b2); FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].b2[i]);
   sz=ArraySize(gDQN[symIdx].W3); FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].W3[i]);
   sz=ArraySize(gDQN[symIdx].b3); FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].b3[i]);
   sz=ArraySize(gDQN[symIdx].feat_mean); FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].feat_mean[i]);
   sz=ArraySize(gDQN[symIdx].feat_std);  FileWriteInteger(handle,sz); for(int i=0;i<sz;i++) FileWriteDouble(handle,gDQN[symIdx].feat_std[i]);

   FileClose(handle);
   return true;
}

bool LoadDQNForSymbol(int symIdx, string filename)
{
   if(!FileIsExist(filename)) return false;

   int handle=FileOpen(filename, FILE_READ|FILE_BIN);
   if(handle==INVALID_HANDLE)
   {
      Print("Error opening DQN file for reading: ",filename," err=",GetLastError());
      return false;
   }

   int version=FileReadInteger(handle);
   if(version!=2)
   {
      FileClose(handle);
      return false;
   }

   gDQN[symIdx].input_dim   = FileReadInteger(handle);
   gDQN[symIdx].hidden1_dim = FileReadInteger(handle);
   gDQN[symIdx].hidden2_dim = FileReadInteger(handle);
   gDQN[symIdx].output_dim  = FileReadInteger(handle);

   int sz;
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].W1,sz); for(int i=0;i<sz;i++) gDQN[symIdx].W1[i]=FileReadDouble(handle);
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].b1,sz); for(int i=0;i<sz;i++) gDQN[symIdx].b1[i]=FileReadDouble(handle);
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].W2,sz); for(int i=0;i<sz;i++) gDQN[symIdx].W2[i]=FileReadDouble(handle);
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].b2,sz); for(int i=0;i<sz;i++) gDQN[symIdx].b2[i]=FileReadDouble(handle);
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].W3,sz); for(int i=0;i<sz;i++) gDQN[symIdx].W3[i]=FileReadDouble(handle);
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].b3,sz); for(int i=0;i<sz;i++) gDQN[symIdx].b3[i]=FileReadDouble(handle);
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].feat_mean,sz); for(int i=0;i<sz;i++) gDQN[symIdx].feat_mean[i]=FileReadDouble(handle);
   sz=FileReadInteger(handle); ArrayResize(gDQN[symIdx].feat_std,sz);  for(int i=0;i<sz;i++) gDQN[symIdx].feat_std[i]=FileReadDouble(handle);

   FileClose(handle);

   if(VerboseLogging) Print("DQN loaded for ",gSymbols[symIdx]," from ",filename);
   return true;
}

void InitOrRandomizeDQN(int symIdx)
{
   gDQN[symIdx].input_dim   = (StateDimension>0 ? StateDimension : 8);
   gDQN[symIdx].hidden1_dim = (HiddenSize>0 ? HiddenSize : 8);
   gDQN[symIdx].hidden2_dim = (HiddenSize2>0 ? HiddenSize2 : gDQN[symIdx].hidden1_dim);
   gDQN[symIdx].output_dim  = (ActionCount>0 ? ActionCount : 3);

   int inDim=gDQN[symIdx].input_dim;
   int h1Dim=gDQN[symIdx].hidden1_dim;
   int h2Dim=gDQN[symIdx].hidden2_dim;
   int outDim=gDQN[symIdx].output_dim;

   ArrayResize(gDQN[symIdx].W1, h1Dim*inDim);
   ArrayResize(gDQN[symIdx].b1, h1Dim);
   ArrayResize(gDQN[symIdx].W2, h2Dim*h1Dim);
   ArrayResize(gDQN[symIdx].b2, h2Dim);
   ArrayResize(gDQN[symIdx].W3, outDim*h2Dim);
   ArrayResize(gDQN[symIdx].b3, outDim);
   ArrayResize(gDQN[symIdx].feat_mean,inDim);
   ArrayResize(gDQN[symIdx].feat_std,inDim);

   double s1=1.0/MathSqrt((double)inDim);
   double s2=1.0/MathSqrt((double)h1Dim);
   double s3=1.0/MathSqrt((double)h2Dim);

   for(int i=0;i<ArraySize(gDQN[symIdx].W1);i++){ double r=(double)MathRand()/32767.0; gDQN[symIdx].W1[i]=(r*2.0-1.0)*s1; }
   for(int i=0;i<h1Dim;i++) gDQN[symIdx].b1[i]=0.0;

   for(int i=0;i<ArraySize(gDQN[symIdx].W2);i++){ double r=(double)MathRand()/32767.0; gDQN[symIdx].W2[i]=(r*2.0-1.0)*s2; }
   for(int i=0;i<h2Dim;i++) gDQN[symIdx].b2[i]=0.0;

   for(int i=0;i<ArraySize(gDQN[symIdx].W3);i++){ double r=(double)MathRand()/32767.0; gDQN[symIdx].W3[i]=(r*2.0-1.0)*s3; }
   for(int i=0;i<outDim;i++) gDQN[symIdx].b3[i]=0.0;

   for(int i=0;i<inDim;i++){ gDQN[symIdx].feat_mean[i]=0.0; gDQN[symIdx].feat_std[i]=1.0; }
}

//==================================================================
//  ZONE DETECTION / UPDATE
//==================================================================
void DetectZones(string symbol)
{
   int bars=iBars(symbol, BaseTF);
   if(bars<=0) return;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   int minBase=MathMax(1,MinZoneBaseBars);
   int maxBase=MathMax(minBase,MaxZoneBaseBars);
   if(bars<maxBase+2) return;

   int breakoutShift=1;
   datetime lastClosedTime=iTime(symbol, BaseTF, breakoutShift);

   bool newZoneFound=false;
   double newHigh=0.0,newLow=0.0;
   bool newIsDemand=false;
   int newStartIndex=-1;

   for(int baseLen=minBase;baseLen<=maxBase;baseLen++)
   {
      int startIndex=breakoutShift+baseLen;
      if(startIndex>=bars) continue;

      bool consolidated=true;
      double highPrice=iHigh(symbol, BaseTF, startIndex);
      double lowPrice =iLow (symbol, BaseTF, startIndex);

      for(int iBar=startIndex-1;iBar>breakoutShift;iBar--)
      {
         double h=iHigh(symbol, BaseTF, iBar);
         double l=iLow (symbol, BaseTF, iBar);
         if(h>highPrice) highPrice=h;
         if(l<lowPrice)  lowPrice=l;

         if((highPrice-lowPrice) > ZoneMaxHeightPoints*point){ consolidated=false; break; }
      }
      if(!consolidated) continue;

      double closePrice=iClose(symbol, BaseTF, breakoutShift);
      double breakoutLow=iLow(symbol, BaseTF, breakoutShift);
      double breakoutHigh=iHigh(symbol, BaseTF, breakoutShift);

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
            if(MathAbs(zones[j].high-highPrice)<point && MathAbs(zones[j].low-lowPrice)<point)
               { duplicate=true; break; }
         }
      }
      if(overlaps||duplicate) continue;

      newZoneFound=true;
      newHigh=highPrice;
      newLow=lowPrice;
      newIsDemand=isDemand;
      newStartIndex=startIndex;
      break;
   }

   if(!newZoneFound) return;

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
   zones[zc].high=newHigh;
   zones[zc].low=newLow;
   zones[zc].startTime=iTime(symbol, BaseTF, (newStartIndex>0?newStartIndex:breakoutShift+MinZoneBaseBars));
   zones[zc].endTime=TimeCurrent() + (datetime)PeriodSeconds(BaseTF)*ZoneExtendBars;
   zones[zc].breakoutTime=lastClosedTime;
   zones[zc].isDemand=newIsDemand;
   zones[zc].tested=false;
   zones[zc].broken=false;
   zones[zc].name="SDZone_"+symbol+"_"+IntegerToString(zc)+"_"+TimeToString(zones[zc].startTime,TIME_DATE|TIME_SECONDS);
}

void UpdateZones(string symbol)
{
   int total=ArraySize(zones);
   if(total<=0) return;

   double prevHigh=iHigh(symbol, BaseTF, 1);
   double prevLow =iLow (symbol, BaseTF, 1);
   double prevClose=iClose(symbol, BaseTF, 1);
   datetime lastClosedTime=iTime(symbol, BaseTF, 1);

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

      if(zones[i].isDemand){ if(prevClose<zones[i].low) zones[i].broken=true; }
      else{ if(prevClose>zones[i].high) zones[i].broken=true; }

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

         datetime midTime=zones[i].startTime + (zones[i].endTime-zones[i].startTime)/2;
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

//==================================================================
//  POSITION COUNT / LOTS / OPEN / GRID / CORRELATION
//==================================================================
void CountOpenPositions()
{
   for(int i=0;i<gSymbolCount;i++){ gPositionsCount[i]=0; gTrades[i].Clear(); }

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym=PositionGetString(POSITION_SYMBOL);
      long mg=PositionGetInteger(POSITION_MAGIC);
      double price=PositionGetDouble(POSITION_PRICE_OPEN);

      int idx=SymbolIndex(sym);
      if(idx<0) continue;
      if((int)mg!=gMagics[idx]) continue;

      gPositionsCount[idx]++;
      gTrades[idx].Add(price);
   }
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

double CalculateLot(int positionIndex, string symbol)
{
   double scale=GetEquityBudgetScale();
   double raw=Lots * scale * MathPow(LotExponent, positionIndex);
   return NormalizeVolume(symbol, raw);
}

bool OpenPosition(string symbol, ENUM_ORDER_TYPE orderType, double volume, int stopLossPoints, int takeProfitPoints, int magic)
{
   trade.SetExpertMagicNumber(magic);
   trade.SetDeviationInPoints(Slippage);

   double vol=NormalizeVolume(symbol, volume);
   if(vol<=0.0) return false;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double price=0.0, sl=0.0, tp=0.0;

   if(orderType==ORDER_TYPE_BUY)
   {
      price=SymbolInfoDouble(symbol,SYMBOL_ASK);
      if(stopLossPoints>0)   sl=price - stopLossPoints*point;
      if(takeProfitPoints>0) tp=price + takeProfitPoints*point;
   }
   else
   {
      price=SymbolInfoDouble(symbol,SYMBOL_BID);
      if(stopLossPoints>0)   sl=price + stopLossPoints*point;
      if(takeProfitPoints>0) tp=price - takeProfitPoints*point;
   }

   // stops level check (important for CFDs/metals)
   if(!CheckStopsDistanceOk(symbol, price, sl, tp))
   {
      // if stops invalid, try sending without SL/TP
      sl=0.0; tp=0.0;
   }

   return trade.PositionOpen(symbol, orderType, vol, price, sl, tp);
}

ENUM_POSITION_TYPE GetPositionType(string symbol, int magic)
{
   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)==symbol && (int)PositionGetInteger(POSITION_MAGIC)==magic)
         return (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   }
   return POSITION_TYPE_BUY;
}

void CalculateLevels(string symbol, int positionsCount, CArrayDouble &trades,
                     int magic, double &buyLevel, double &sellLevel)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   double range=(double)DefaultPips * point;

   if(UseDynamicPips)
   {
      double high=-DBL_MAX, low=DBL_MAX;
      int bars=iBars(symbol, BaseTF);
      int n=MathMin(Depth, bars-1);
      if(n<2) n=2;

      for(int i=1;i<=n;i++)
      {
         double h=iHigh(symbol, BaseTF, i);
         double l=iLow (symbol, BaseTF, i);
         if(h>high) high=h;
         if(l<low)  low=l;
      }
      if(high>low && PipsFactor>0.0)
         range=(high-low)/PipsFactor;
   }

   double d1DistNorm=0.0,d1SlopeNorm=0.0,d1SideDurNorm=0.0;
   GetD1TrendFeatures(symbol,d1DistNorm,d1SlopeNorm,d1SideDurNorm);

   if(positionsCount>0 && trades.Total()>0)
   {
      double avg=0.0;
      for(int i=0;i<trades.Total();i++) avg+=trades.At(i);
      avg/=trades.Total();

      ENUM_POSITION_TYPE pt=GetPositionType(symbol,magic);

      int d1TrendDir=0;
      if(d1DistNorm>0.0) d1TrendDir=1;
      else if(d1DistNorm<0.0) d1TrendDir=-1;

      bool fadingUptrend=(pt==POSITION_TYPE_SELL && d1TrendDir==1);
      bool fadingDowntrend=(pt==POSITION_TYPE_BUY && d1TrendDir==-1);

      if(fadingUptrend || fadingDowntrend)
      {
         double trendStrength = MathAbs(d1DistNorm)/MathMax(1e-9,D1_StrongTrendDistance) + 0.5*MathAbs(d1SlopeNorm);
         trendStrength=Clamp(trendStrength,0.0,1.0);
         if(d1SideDurNorm>trendStrength) trendStrength=d1SideDurNorm;
         double widenFactor = 1.0 + (D1_MaxGridWidenFactor-1.0)*trendStrength;
         range *= widenFactor;
      }

      if(pt==POSITION_TYPE_BUY){ buyLevel=avg-range; sellLevel=0.0; }
      else{ sellLevel=avg+range; buyLevel=0.0; }
   }
   else
   {
      double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
      double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
      buyLevel=bid-range;
      sellLevel=ask+range;
   }
}

double CalculateCorrelation(double &a[], double &b[], int size)
{
   if(size<=1) return 0.0;
   double sumX=0,sumY=0,sumXX=0,sumYY=0,sumXY=0;
   for(int i=0;i<size;i++)
   {
      sumX+=a[i]; sumY+=b[i];
      sumXX+=a[i]*a[i]; sumYY+=b[i]*b[i];
      sumXY+=a[i]*b[i];
   }
   double num=size*sumXY - sumX*sumY;
   double denL=size*sumXX - sumX*sumX;
   double denR=size*sumYY - sumY*sumY;
   double den=MathSqrt(denL*denR);
   if(den<=1e-12) return 0.0;
   return num/den;
}

bool CheckCorrelation()
{
   if(!UseCorrelation) return true;
   if(gSymbolCount<2) return true;

   string sym1=gSymbols[0];
   string sym2=gSymbols[1];

   datetime bt=iTime(sym1, BaseTF, 0);
   if(bt!=gLastCorrBarTime)
   {
      gLastCorrBarTime=bt;

      double p1[], p2[];
      ArrayResize(p1,CorrelationPeriod);
      ArrayResize(p2,CorrelationPeriod);
      for(int i=0;i<CorrelationPeriod;i++)
      {
         p1[i]=iClose(sym1,BaseTF,i);
         p2[i]=iClose(sym2,BaseTF,i);
      }
      gLastCorrelation=CalculateCorrelation(p1,p2,CorrelationPeriod);
      if(VerboseLogging) Print("Correlation ",sym1,"/",sym2," = ",DoubleToString(gLastCorrelation,3));
   }

   return (MathAbs(gLastCorrelation) >= MinCorrelation);
}

//==================================================================
//  CLOSE / EQUITY STOP / REWARD
//==================================================================
bool ClosePositions(string symbol,int magic)
{
   bool success=false;
   double closedProfit=0.0;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)==symbol && (int)PositionGetInteger(POSITION_MAGIC)==magic)
      {
         double p=PositionGetDouble(POSITION_PROFIT);
         if(trade.PositionClose(ticket))
         {
            success=true;
            closedProfit+=p;
         }
      }
   }

   if(success) gEAClosedProfit += closedProfit;
   return success;
}

void CloseAllPositions()
{
   for(int i=0;i<gSymbolCount;i++)
   {
      ClosePositions(gSymbols[i], gMagics[i]);
      gFirstTradeTime[i]=0;
   }
}

bool CheckEquityStop()
{
   double eq=GetEAEquity();
   if(maxEquity<=1e-9) return false;
   return (eq < maxEquity*(1.0-EquityRiskPercent/100.0));
}

double ComputeEquityShapingReward()
{
   double eq=GetEAEquity();
   double bal=gEAStartEquity;

   if(gLastEquityForReward<=0.0) gLastEquityForReward=eq;

   double deltaFrac=0.0;
   if(bal>1e-9) deltaFrac=(eq-gLastEquityForReward)/bal;

   gLastEquityForReward=eq;

   double dd=0.0;
   if(maxEquity>1e-9 && eq<maxEquity) dd=(maxEquity-eq)/maxEquity;

   double reward = EquityRewardScale*deltaFrac - DrawdownPenaltyScale*dd;
   return Clamp(reward,-5.0,5.0);
}

//==================================================================
//  CCI EXIT / TP / TRAILING
//==================================================================
bool CheckCCIExit()
{
   if(!UseCCI) return false;
   bool acted=false;

   for(int i=0;i<gSymbolCount;i++)
   {
      if(gPositionsCount[i]<=0) continue;

      string sym=gSymbols[i];
      int magic=gMagics[i];

      double cciVal=GetCCIValue(sym, BaseTF, CCI_Period, 0);
      ENUM_POSITION_TYPE pt=GetPositionType(sym,magic);

      if((pt==POSITION_TYPE_BUY && cciVal < -CCI_Level) ||
         (pt==POSITION_TYPE_SELL && cciVal >  CCI_Level))
      {
         if(VerboseLogging) Print("CCI exit on ",sym);
         ClosePositions(sym,magic);
         acted=true;
      }
   }
   return acted;
}

void CheckPairTakeProfit(string symbol, CArrayDouble &trades, int magic)
{
   if(trades.Total()==0) return;

   double avg=0.0;
   for(int i=0;i<trades.Total();i++) avg += trades.At(i);
   avg /= (double)trades.Total();

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   ENUM_POSITION_TYPE pt=GetPositionType(symbol,magic);

   if(pt==POSITION_TYPE_BUY && bid >= avg + TakeProfit*point)
      ClosePositions(symbol,magic);
   else if(pt==POSITION_TYPE_SELL && ask <= avg - TakeProfit*point)
      ClosePositions(symbol,magic);
}

void CheckTakeProfit()
{
   for(int i=0;i<gSymbolCount;i++)
      if(gPositionsCount[i]>0)
         CheckPairTakeProfit(gSymbols[i], gTrades[i], gMagics[i]);
}

void TrailingStopForPair(string symbol, CArrayDouble &trades, int magic)
{
   if(trades.Total()==0) return;

   double avg=0.0;
   for(int i=0;i<trades.Total();i++) avg+=trades.At(i);
   avg/=trades.Total();

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   ENUM_POSITION_TYPE pt=GetPositionType(symbol,magic);

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)!=symbol || (int)PositionGetInteger(POSITION_MAGIC)!=magic)
         continue;

      double sl=PositionGetDouble(POSITION_SL);

      if(pt==POSITION_TYPE_BUY)
      {
         if(bid-avg > TrailStart*point)
         {
            double newSL=bid - TrailStop*point;
            if(newSL>sl || sl==0.0) trade.PositionModify(ticket,newSL,0.0);
         }
      }
      else
      {
         if(avg-ask > TrailStart*point)
         {
            double newSL=ask + TrailStop*point;
            if(newSL<sl || sl==0.0) trade.PositionModify(ticket,newSL,0.0);
         }
      }
   }
}

void TrailingStop()
{
   if(!UseTrailingStop) return;
   for(int i=0;i<gSymbolCount;i++)
      TrailingStopForPair(gSymbols[i], gTrades[i], gMagics[i]);
}

double CalculatePositionsPnL(string symbol,int magic)
{
   double total=0.0;
   int n=PositionsTotal();
   for(int i=0;i<n;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)==symbol && (int)PositionGetInteger(POSITION_MAGIC)==magic)
         total += PositionGetDouble(POSITION_PROFIT);
   }
   return total;
}

//==================================================================
//  CLASSIC GRID (non-DQN mode)
//==================================================================
void ManagePair(string symbol,int &positionsCount,CArrayDouble &trades,int magic,datetime &firstTradeTime)
{
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);

   double buyLevel=0.0,sellLevel=0.0;
   CalculateLevels(symbol,positionsCount,trades,magic,buyLevel,sellLevel);

   double rsiValue=GetRSIValue(symbol, BaseTF, RSI_Period, 0);

   if(positionsCount==0)
   {
      if(rsiValue <= RSI_Minimum)
      {
         if(OpenPosition(symbol, ORDER_TYPE_BUY, CalculateLot(0,symbol), StopLoss, TakeProfit, magic))
            firstTradeTime=TimeCurrent();
      }
      else if(rsiValue >= RSI_Maximum)
      {
         if(OpenPosition(symbol, ORDER_TYPE_SELL, CalculateLot(0,symbol), StopLoss, TakeProfit, magic))
            firstTradeTime=TimeCurrent();
      }
   }
   else
   {
      if(positionsCount>=MaxTrades) return;

      ENUM_POSITION_TYPE pt=GetPositionType(symbol,magic);

      if(pt==POSITION_TYPE_BUY)
      {
         if(bid<=buyLevel && rsiValue<=RSI_Minimum)
            OpenPosition(symbol, ORDER_TYPE_BUY, CalculateLot(positionsCount,symbol), StopLoss, TakeProfit, magic);
      }
      else
      {
         if(ask>=sellLevel && rsiValue>=RSI_Maximum)
            OpenPosition(symbol, ORDER_TYPE_SELL, CalculateLot(positionsCount,symbol), StopLoss, TakeProfit, magic);
      }
   }
}

//==================================================================
//  DQN-BASED GRID MANAGEMENT
//==================================================================
void ManagePairWithDQN(string symbol,int symIdx,int &positionsCount,CArrayDouble &trades,int magic,datetime &firstTradeTime)
{
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double state[];
   GetCurrentState(symbol,positionsCount,trades,state);

   bool extreme=IsExtremeState(state);
   if(SeparateExtremeStates)
   {
      int sz=ArraySize(state);
      if(sz>0) state[sz-1]=(extreme?1.0:0.0);
   }

   // D1 trend from state (if present)
   double d1Dist=0.0, d1Slope=0.0, d1Side=0.0;
   if(ArraySize(state)>11) d1Dist=state[11];
   if(ArraySize(state)>12) d1Slope=state[12];
   if(ArraySize(state)>13) d1Side=state[13];

   double combinedTrend=d1Dist + 0.5*d1Slope;
   double reversalRisk=d1Side;

   // Ensure DQN dims match current state array
   int wantIn=ArraySize(state);
   if(gDQN[symIdx].input_dim != wantIn || gDQN[symIdx].output_dim != ActionCount)
   {
      // Re-init if mismatch
      gDQN[symIdx].input_dim = wantIn;
      InitOrRandomizeDQN(symIdx);
   }

   int action=DQNSelectAction(symIdx,state);

   ENUM_ORDER_TYPE orderType=ORDER_TYPE_BUY;
   bool shouldTrade=false;

   if(action==1){ orderType=ORDER_TYPE_BUY; shouldTrade=true; }
   else if(action==2){ orderType=ORDER_TYPE_SELL; shouldTrade=true; }

   if(shouldTrade)
   {
      if(positionsCount==0)
      {
         if(OpenPosition(symbol, orderType, CalculateLot(0,symbol), StopLoss, TakeProfit, magic))
         {
            firstTradeTime=TimeCurrent();
            if(isTraining)
            {
               double reward=ComputeEquityShapingReward();
               reward=ScaleRewardByRegime(reward,extreme);
               DQNUpdate(symIdx,state,action,reward,state,false);
            }
         }
      }
      else
      {
         if(positionsCount>=MaxTrades) return;

         ENUM_POSITION_TYPE pt=GetPositionType(symbol,magic);

         if((pt==POSITION_TYPE_BUY && orderType==ORDER_TYPE_BUY) ||
            (pt==POSITION_TYPE_SELL && orderType==ORDER_TYPE_SELL))
         {
            double buyLevel=0.0,sellLevel=0.0;
            CalculateLevels(symbol,positionsCount,trades,magic,buyLevel,sellLevel);

            if(pt==POSITION_TYPE_BUY)
            {
               if(bid<=buyLevel)
               {
                  if(OpenPosition(symbol, ORDER_TYPE_BUY, CalculateLot(positionsCount,symbol), StopLoss, TakeProfit, magic))
                  {
                     if(isTraining)
                     {
                        double reward=-1.0;
                        if(combinedTrend < -TrendPenaltyThreshold) reward -= TrendPenaltyFactor*MathAbs(combinedTrend);
                        reward -= TrendPenaltyFactor*reversalRisk;
                        reward += ComputeEquityShapingReward();
                        reward=ScaleRewardByRegime(reward,extreme);
                        DQNUpdate(symIdx,state,action,reward,state,false);
                     }
                  }
               }
            }
            else
            {
               if(ask>=sellLevel)
               {
                  if(OpenPosition(symbol, ORDER_TYPE_SELL, CalculateLot(positionsCount,symbol), StopLoss, TakeProfit, magic))
                  {
                     if(isTraining)
                     {
                        double reward=-1.0;
                        if(combinedTrend > TrendPenaltyThreshold) reward -= TrendPenaltyFactor*MathAbs(combinedTrend);
                        reward -= TrendPenaltyFactor*reversalRisk;
                        reward += ComputeEquityShapingReward();
                        reward=ScaleRewardByRegime(reward,extreme);
                        DQNUpdate(symIdx,state,action,reward,state,false);
                     }
                  }
               }
            }
         }
      }
   }

   // basket close heuristic
   double profit=CalculatePositionsPnL(symbol,magic);

   if(profit>0.0 && positionsCount>0)
   {
      bool shouldClose=false;
      if(profit >= TakeProfit*point*positionsCount*Lots) shouldClose=true;

      if(shouldClose)
      {
         ENUM_POSITION_TYPE pt=GetPositionType(symbol,magic);
         if(ClosePositions(symbol,magic))
         {
            firstTradeTime=0;

            if(isTraining)
            {
               double reward=profit;

               if(pt==POSITION_TYPE_BUY && combinedTrend > TrendPenaltyThreshold) reward += TrendPenaltyFactor*MathAbs(combinedTrend);
               else if(pt==POSITION_TYPE_SELL && combinedTrend < -TrendPenaltyThreshold) reward += TrendPenaltyFactor*MathAbs(combinedTrend);

               reward += ComputeEquityShapingReward();
               reward=ScaleRewardByRegime(reward,extreme);

               DQNUpdate(symIdx,state,action,reward,state,true);

               episodeCount++;
               totalReward += reward;

               if(VerboseLogging)
                  Print("Episode ",episodeCount," finished on ",symbol,
                        " reward=",DoubleToString(reward,3),
                        " avg=",DoubleToString(totalReward/MathMax(1,episodeCount),3));
            }
         }
      }
   }
}

//==================================================================
//  LEARNING DASHBOARD
//==================================================================
string DashboardActionName(int action)
{
   if(action==1) return "BUY";
   if(action==2) return "SELL";
   return "HOLD";
}

int GetDashboardSymbolIndex()
{
   if(gSymbolCount<=0)
      return -1;

   // Explicit dashboard symbol if supplied
   if(StringLen(DashboardSymbol)>0)
   {
      int idx=SymbolIndex(DashboardSymbol);
      if(idx>=0)
         return idx;
   }

   // Prefer the chart symbol
   int chartIdx=SymbolIndex(_Symbol);
   if(chartIdx>=0)
      return chartIdx;

   // Otherwise use the first symbol in the active universe
   return 0;
}

void UpdateLearningDashboard()
{
   if(!ShowLearningDashboard)
   {
      Comment("");
      return;
   }

   // Refresh positions so the monitor reflects any order opened/closed
   // earlier in the current tick.
   CountOpenPositions();

   int symIdx=GetDashboardSymbolIndex();
   if(symIdx<0 || symIdx>=gSymbolCount)
      return;

   string symbol=gSymbols[symIdx];

   // Build the current state for contextual diagnostics such as ATR.
   double state[];
   GetCurrentState(
      symbol,
      gPositionsCount[symIdx],
      gTrades[symIdx],
      state
   );

   // Display the Q-values that existed when the most recent action was
   // selected, not values recomputed after a possible training update.
   double qHold=gLastQHold[symIdx];
   double qBuy =gLastQBuy[symIdx];
   double qSell=gLastQSell[symIdx];

   // Equity and drawdown
   double equity=GetEAEquity();

   double ddPct=0.0;
   if(maxEquity>1e-9 && equity<maxEquity)
      ddPct=100.0*(maxEquity-equity)/maxEquity;

   // Reward statistics
   double avgReward=0.0;
   if(episodeCount>0)
      avgReward=totalReward/(double)episodeCount;

   // Latest selected action
   int action=gLastSelectedAction[symIdx];
   string actionName=DashboardActionName(action);

   string actionSource=
      (gLastActionExploratory[symIdx]
       ? "EXPLORATION"
       : "POLICY");

   string actionLine;
   if(gHasActionDecision[symIdx])
      actionLine=actionName + " [" + actionSource + "]";
   else
      actionLine="WAITING FOR FIRST DECISION";

   string mode=
      (isTraining ? "TRAINING" : "INFERENCE");

   // ATR ratio is state feature 6 in the public learning edition when present.
   double atrRatio=1.0;
   if(ArraySize(state)>6)
      atrRatio=state[6];

   string dashboard="";

   dashboard += "ADAPTIVE DQN - LEARNING MONITOR\n";
   dashboard += "----------------------------------------\n";

   dashboard +=
      "Symbol: " + symbol +
      "   TF: " + EnumToString(BaseTF) + "\n";

   dashboard +=
      "Mode: " + mode +
      "   Epsilon: " +
      DoubleToString(currentEpsilon,4) + "\n";

   dashboard += "----------------------------------------\n";

   dashboard += "Q(HOLD): " + DoubleToString(qHold,4) + "\n";
   dashboard += "Q(BUY) : " + DoubleToString(qBuy,4) + "\n";
   dashboard += "Q(SELL): " + DoubleToString(qSell,4) + "\n";

   dashboard +=
      "\nAction: " +
      actionLine + "\n";

   dashboard += "----------------------------------------\n";

   dashboard +=
      "Episodes: " +
      IntegerToString(episodeCount) + "\n";

   dashboard +=
      "Total Reward: " +
      DoubleToString(totalReward,3) + "\n";

   dashboard +=
      "Avg Reward: " +
      DoubleToString(avgReward,3) + "\n";

   dashboard += "----------------------------------------\n";

   dashboard +=
      "EA Equity: $" +
      DoubleToString(equity,2) + "\n";

   dashboard +=
      "Drawdown: " +
      DoubleToString(ddPct,2) + "%\n";

   dashboard +=
      "Open Positions: " +
      IntegerToString(gPositionsCount[symIdx]) + "\n";

   dashboard +=
      "ATR Ratio: " +
      DoubleToString(atrRatio,3) + "\n";

   Comment(dashboard);
}

//==================================================================
//  INIT / DEINIT
//==================================================================
int OnInit()
{
   trade.SetDeviationInPoints(Slippage);
   MathSrand((int)GetTickCount());

   BuildSymbolUniverse();
   if(gSymbolCount<=0)
   {
      Print("No tradable symbols selected. Check SymbolMode/SymbolsList/MarketWatch.");
      return INIT_FAILED;
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      gPositionsCount[i]=0;
      gTrades[i].Clear();
      gFirstTradeTime[i]=0;

      // Dashboard state
      gLastSelectedAction[i]=0;
      gLastActionExploratory[i]=false;
      gHasActionDecision[i]=false;
      gLastQHold[i]=0.0;
      gLastQBuy[i]=0.0;
      gLastQSell[i]=0.0;

      gLastBarTime[i]=0;
      gD1LastBarTime[i]=0;
      gD1DistCache[i]=0.0; gD1SlopeCache[i]=0.0; gD1SideDurCache[i]=0.0;

      gHTFLastBarTime[i]=0;
      gHTFCCICache[i]=0.5;
      gHTFATRCache[i]=0.0;

      gATRLastBarTime[i]=0;
      gATRratioCache[i]=1.0;

      // Load/init DQN per symbol
      string eaName=MQLInfoString(MQL_PROGRAM_NAME);
      string base=(StringLen(QTablePrefix)>0 ? QTablePrefix : eaName+"_DQN_");
      string fname=base + gSymbols[i] + ".dat";

      if(!LoadDQNForSymbol(i,fname))
      {
         if(VerboseLogging) Print("Initializing new DQN for ",gSymbols[i]);
         InitOrRandomizeDQN(i);
      }
      else
      {
         // if file loaded but dims not aligned with current settings, reinit
         if(gDQN[i].output_dim != ActionCount) InitOrRandomizeDQN(i);
      }
   }

   gEAStartEquity = (EquityBudget>1e-9 ? EquityBudget : AccountInfoDouble(ACCOUNT_BALANCE));
   if(gEAStartEquity<=1e-9) gEAStartEquity=1000.0;

   gEAClosedProfit=0.0;
   maxEquity=gEAStartEquity;
   gLastEquityForReward=gEAStartEquity;

   isTraining=TrainingMode;
   tickCounter=0;
   totalReward=0.0;
   episodeCount=0;
   currentEpsilon=(isTraining ? ExplorationRate : MinExplorationRate);

   gLastCorrelation=1.0;
   gLastCorrBarTime=0;

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   Comment("");

   if(UseDQN && SaveQTable)
   {
      string eaName=MQLInfoString(MQL_PROGRAM_NAME);
      string base=(StringLen(QTablePrefix)>0 ? QTablePrefix : eaName+"_DQN_");

      for(int i=0;i<gSymbolCount;i++)
      {
         string fname=base + gSymbols[i] + ".dat";
         if(SaveDQNForSymbol(i,fname))
         {
            if(VerboseLogging) Print("DQN saved for ",gSymbols[i]," to ",fname);
         }
         else
         {
            Print("Error saving DQN for ",gSymbols[i]," to ",fname);
         }
      }
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
}

//==================================================================
//  MAIN TICK
//==================================================================
void OnTick()
{
   double equity=GetEAEquity();
   if(equity>maxEquity) maxEquity=equity;

   if(UseEquityStop && CheckEquityStop())
   {
      if(VerboseLogging) Print("Equity stop triggered (virtual EA equity). Closing all.");
      CloseAllPositions();
      UpdateLearningDashboard();
      return;
   }

   CountOpenPositions();

   // zones update on BaseTF closed bar per symbol
   for(int i=0;i<gSymbolCount;i++)
   {
      string sym=gSymbols[i];
      datetime lastClosed=iTime(sym, BaseTF, 1);
      if(lastClosed!=gLastBarTime[i])
      {
         gLastBarTime[i]=lastClosed;
         DetectZones(sym);
         UpdateZones(sym);
      }
   }

   if(CheckCCIExit())
   {
      UpdateLearningDashboard();
      return;
   }

   TrailingStop();
   CheckTakeProfit();

   if(UseDQN)
   {
      tickCounter++;
      if(isTraining && TrainingFreq>0 && (tickCounter % TrainingFreq)==0)
      {
         if(currentEpsilon>MinExplorationRate) currentEpsilon*=ExplorationDecay;
         if(currentEpsilon<MinExplorationRate) currentEpsilon=MinExplorationRate;
      }

      for(int i=0;i<gSymbolCount;i++)
      {
         ManagePairWithDQN(gSymbols[i], i, gPositionsCount[i], gTrades[i], gMagics[i], gFirstTradeTime[i]);
      }
   }
   else
   {
      if(UseCorrelation && !CheckCorrelation())
      {
         if(VerboseLogging) Print("Correlation condition failed. No new positions.");
         UpdateLearningDashboard();
         return;
      }

      for(int i=0;i<gSymbolCount;i++)
      {
         ManagePair(gSymbols[i], gPositionsCount[i], gTrades[i], gMagics[i], gFirstTradeTime[i]);
      }
   }

   UpdateLearningDashboard();
}
//+------------------------------------------------------------------+
