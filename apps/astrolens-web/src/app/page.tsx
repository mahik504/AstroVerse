"use client"
import React, { useState, useEffect } from 'react';
import { Upload, Activity, Database, FileText, Satellite, Info, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceArea } from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ConfidenceRouting {
  CNN_Shape: number;
  Transformer_Rhythm: number;
  Physics_MLP: number;
}

interface PredictResult {
  is_exoplanet: boolean;
  probability: number;
  confidence_routing: ConfidenceRouting;
}

interface Target {
  tic_id: string;
  temperature: number;
  mass: number;
  radius?: number;
  distance_pc: number;
}

interface LightcurvePoint {
  phase: number;
  flux: number;
}

export default function AstroLensHome() {
  const [activeTab, setActiveTab] = useState('workspace');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [targets, setTargets] = useState<Target[]>([]);
  const [lightcurveData, setLightcurveData] = useState<LightcurvePoint[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/targets`)
      .then(res => res.json())
      .then(data => setTargets(data))
      .catch(err => console.error("API Error", err));
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setResult(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error('API Error');
      const data = await response.json();
      setResult(data);
      
      const lcRes = await fetch(`${API_URL}/lightcurve/DEMO`);
      const lcData = await lcRes.json();
      
      const chartData = lcData.phase.map((p: number, i: number) => ({
        phase: p,
        flux: lcData.flux[i]
      }));
      setLightcurveData(chartData);
      
    } catch (error) {
      console.error(error);
      alert("Error contacting EvoNex API. Ensure backend is running.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;
    const formData = new FormData();
    formData.append('tic_id', selectedFile?.name.replace('.csv', '') || 'TARGET');
    formData.append('probability', result.probability.toString());
    
    try {
      const response = await fetch(`${API_URL}/report`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      
      const blob = new Blob([data.markdown_content], { type: 'text/markdown' });
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = `Analysis_${selectedFile?.name}.md`;
      a.click();
      URL.revokeObjectURL(blobUrl);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-blue-600 flex flex-col">
      
      {/* Top Navigation - Strict Aerospace */}
      <nav className="border-b border-zinc-800 bg-black">
        <div className="px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Satellite className="w-5 h-5 text-blue-500" />
            <span className="text-sm font-bold tracking-widest uppercase">AstroVerse <span className="text-zinc-500 font-normal ml-2">v2.0.1</span></span>
          </div>
          <div className="flex items-center gap-6 text-xs font-mono uppercase tracking-widest">
            <button 
              onClick={() => setActiveTab('workspace')}
              className={`transition-colors ${activeTab === 'workspace' ? 'text-white border-b-2 border-blue-500 pb-1' : 'text-zinc-500 hover:text-white pb-1'}`}
            >
              Detection Workspace
            </button>
            <button 
              onClick={() => setActiveTab('targets')}
              className={`transition-colors ${activeTab === 'targets' ? 'text-white border-b-2 border-blue-500 pb-1' : 'text-zinc-500 hover:text-white pb-1'}`}
            >
              Target Database
            </button>
            <a href="docs/Model_Card.md" target="_blank" className="text-zinc-500 hover:text-white pb-1">
              Methodology
            </a>
          </div>
        </div>
      </nav>

      {/* Main App Workspace */}
      <div className="flex-1 p-6 max-w-[1600px] mx-auto w-full">
        
        {activeTab === 'targets' && (
          <div className="space-y-4 animate-in fade-in duration-300">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-bold tracking-tight">STScI Exoplanet Candidate Target List (xCTL v08.01)</h2>
              <span className="text-xs font-mono bg-zinc-900 px-2 py-1 border border-zinc-800">{targets.length} Records Active</span>
            </div>
            <Card className="bg-black border-zinc-800 rounded-none">
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left font-mono">
                    <thead className="text-xs text-zinc-400 uppercase bg-zinc-900/50 border-b border-zinc-800">
                      <tr>
                        <th className="px-6 py-3 font-normal">TIC ID</th>
                        <th className="px-6 py-3 font-normal">T_eff (K)</th>
                        <th className="px-6 py-3 font-normal">Mass (M_sun)</th>
                        <th className="px-6 py-3 font-normal">Radius (R_sun)</th>
                        <th className="px-6 py-3 font-normal">Distance (pc)</th>
                        <th className="px-6 py-3 font-normal">Data Source</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-900">
                      {targets.map((target) => (
                        <tr key={target.tic_id} className="hover:bg-zinc-900/30 transition-colors">
                          <td className="px-6 py-3 text-blue-400">{target.tic_id}</td>
                          <td className="px-6 py-3 text-zinc-300">{target.temperature}</td>
                          <td className="px-6 py-3 text-zinc-300">{target.mass}</td>
                          <td className="px-6 py-3 text-zinc-300">{target.radius || 1.0}</td>
                          <td className="px-6 py-3 text-zinc-300">{target.distance_pc}</td>
                          <td className="px-6 py-3 text-zinc-500">MAST Archive</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === 'workspace' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full animate-in fade-in duration-300">
            
            {/* Left Column - Workflow */}
            <div className="lg:col-span-3 space-y-6">
              
              <div className="border border-zinc-800 bg-zinc-950/50 p-5">
                <h3 className="text-xs font-bold uppercase tracking-widest text-zinc-400 mb-4 flex items-center gap-2">
                  <Database className="w-4 h-4" /> Telemetry Ingestion
                </h3>
                
                <label className="border border-dashed border-zinc-700 bg-black p-6 flex flex-col items-center text-center cursor-pointer hover:border-blue-500 transition-colors">
                  <Upload className="w-5 h-5 text-zinc-500 mb-2" />
                  <span className="text-sm font-medium mb-1">
                    {selectedFile ? selectedFile.name : "Select Lightcurve CSV"}
                  </span>
                  <span className="text-[10px] text-zinc-500 font-mono uppercase">Required: 'flux' column</span>
                  <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
                </label>

                <Button 
                  className="w-full mt-4 bg-white text-black hover:bg-zinc-200 rounded-none font-bold uppercase text-xs tracking-widest h-10" 
                  disabled={!selectedFile || isAnalyzing}
                  onClick={runAnalysis}
                >
                  {isAnalyzing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing</> : 'Run EvoNex Inference'}
                </Button>
              </div>

              {/* Methodology Section */}
              <div className="border border-zinc-800 bg-zinc-950/50 p-5">
                <h3 className="text-xs font-bold uppercase tracking-widest text-zinc-400 mb-3 flex items-center gap-2">
                  <Info className="w-4 h-4" /> Methodology
                </h3>
                <p className="text-xs text-zinc-400 leading-relaxed mb-3">
                  This system utilizes a Mixture-of-Experts (MoE) architecture to process photometric time-series data. 
                </p>
                <ul className="text-xs text-zinc-500 space-y-2 font-mono">
                  <li>&gt; <strong>CNN:</strong> Analyzes transit morphology.</li>
                  <li>&gt; <strong>Transformer:</strong> Detects orbital periodicity.</li>
                  <li>&gt; <strong>MLP:</strong> Enforces physical stellar constraints.</li>
                </ul>
              </div>

            </div>

            {/* Right Column - Analysis */}
            <div className="lg:col-span-9 flex flex-col gap-6">
              
              {/* Chart */}
              <div className="border border-zinc-800 bg-black flex-1 flex flex-col min-h-[400px]">
                <div className="border-b border-zinc-800 px-5 py-3 flex items-center justify-between bg-zinc-950/50">
                  <span className="text-xs font-bold uppercase tracking-widest text-zinc-400">Time-Series Flux Analysis</span>
                  {lightcurveData.length > 0 && <span className="text-[10px] font-mono bg-blue-900/30 text-blue-400 px-2 py-1 border border-blue-900/50 uppercase">Detrended • Normalized</span>}
                </div>
                <div className="p-5 flex-1 w-full">
                  {!result && !isAnalyzing ? (
                    <div className="w-full h-full flex flex-col items-center justify-center text-zinc-600 font-mono text-xs">
                      <Activity className="w-6 h-6 mb-2 opacity-50" />
                      <span>NO TELEMETRY LOADED</span>
                    </div>
                  ) : isAnalyzing ? (
                    <div className="w-full h-full flex flex-col items-center justify-center text-blue-500 font-mono text-xs">
                      <Loader2 className="w-6 h-6 animate-spin mb-3" />
                      <span className="animate-pulse">EXECUTING TENSOR OPERATIONS...</span>
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={lightcurveData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="2 2" stroke="#27272a" vertical={false} />
                        <XAxis dataKey="phase" stroke="#52525b" tick={{fill: '#71717a', fontSize: 10, fontFamily: 'monospace'}} type="number" domain={['dataMin', 'dataMax']} />
                        <YAxis stroke="#52525b" tick={{fill: '#71717a', fontSize: 10, fontFamily: 'monospace'}} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(3)} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#000', borderColor: '#27272a', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }}
                          itemStyle={{ color: '#3b82f6' }}
                        />
                        <ReferenceArea x1={-0.05} x2={0.05} fill="#3b82f6" fillOpacity={0.1} />
                        <Line type="monotone" dataKey="flux" stroke="#ffffff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              {/* Bottom Results Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Result */}
                <div className="border border-zinc-800 bg-black flex flex-col h-48">
                  <div className="border-b border-zinc-800 px-5 py-3 bg-zinc-950/50">
                    <span className="text-xs font-bold uppercase tracking-widest text-zinc-400">Inference Result</span>
                  </div>
                  <div className="p-6 flex flex-col justify-center flex-1">
                    {result ? (
                      <>
                        <div className="flex items-center gap-2 mb-2">
                          {result.is_exoplanet ? <CheckCircle2 className="w-5 h-5 text-emerald-500" /> : <AlertTriangle className="w-5 h-5 text-orange-500" />}
                          <h2 className={`text-xl font-bold uppercase tracking-wide ${result.is_exoplanet ? 'text-emerald-500' : 'text-orange-500'}`}>
                            {result.is_exoplanet ? 'Exoplanet Transit Confirmed' : 'Astrophysical False Positive'}
                          </h2>
                        </div>
                        <div className="flex items-baseline gap-2 mt-2">
                          <span className="text-5xl font-light tracking-tighter text-white">{result.probability}%</span>
                          <span className="text-[10px] text-zinc-500 uppercase font-mono tracking-widest">Confidence Score</span>
                        </div>
                      </>
                    ) : (
                      <span className="text-zinc-600 font-mono text-xs uppercase">Awaiting Inference</span>
                    )}
                  </div>
                </div>

                {/* Explainability Matrix */}
                <div className="border border-zinc-800 bg-black flex flex-col h-48 relative overflow-hidden">
                  <div className="border-b border-zinc-800 px-5 py-3 bg-zinc-950/50 flex justify-between items-center">
                    <span className="text-xs font-bold uppercase tracking-widest text-zinc-400">Expert Routing Matrix</span>
                    {result && (
                      <button onClick={downloadReport} className="text-[10px] font-mono text-blue-400 hover:text-white uppercase tracking-widest flex items-center gap-1">
                        <FileText className="w-3 h-3" /> Download Report
                      </button>
                    )}
                  </div>
                  <div className="p-5 flex-1 flex flex-col justify-center space-y-4">
                    {result ? (
                      <>
                        <div>
                          <div className="flex justify-between font-mono text-[10px] uppercase mb-1">
                            <span className="text-zinc-400">Local Expert (CNN)</span>
                            <span className="text-white">{result.confidence_routing.CNN_Shape}%</span>
                          </div>
                          <Progress value={result.confidence_routing.CNN_Shape} className="h-1 bg-zinc-900 rounded-none" indicatorClassName="bg-white" />
                        </div>
                        <div>
                          <div className="flex justify-between font-mono text-[10px] uppercase mb-1">
                            <span className="text-zinc-400">Global Expert (Transformer)</span>
                            <span className="text-white">{result.confidence_routing.Transformer_Rhythm}%</span>
                          </div>
                          <Progress value={result.confidence_routing.Transformer_Rhythm} className="h-1 bg-zinc-900 rounded-none" indicatorClassName="bg-white" />
                        </div>
                        <div>
                          <div className="flex justify-between font-mono text-[10px] uppercase mb-1">
                            <span className="text-zinc-400">Stellar Expert (Physics MLP)</span>
                            <span className="text-white">{result.confidence_routing.Physics_MLP}%</span>
                          </div>
                          <Progress value={result.confidence_routing.Physics_MLP} className="h-1 bg-zinc-900 rounded-none" indicatorClassName="bg-white" />
                        </div>
                      </>
                    ) : (
                      <span className="text-zinc-600 font-mono text-xs uppercase">Awaiting Inference</span>
                    )}
                  </div>
                </div>

              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}
