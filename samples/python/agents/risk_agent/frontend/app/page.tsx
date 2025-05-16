'use client';

import { Center, Grid, GridItem, Heading, Spinner, Table, Thead, Tbody, Tr, Th, Td, Box, Divider } from "@chakra-ui/react";
import React, { useEffect } from 'react';
import type { Node, Edge } from 'reactflow';
import { Position } from 'reactflow';
import ExecutionGraph from "../components/ExecutionGraph";
import AgentTable from "../components/AgentTable";
import { useAgentState } from "../lib/useAgentState";
import dagre from 'dagre';

function statusColor(status?: string) {
  switch (status) {
    case "supported": return "#38A169";
    case "rejected": return "#E53E3E";
    case "investigating": return "#3182CE";
    default: return "#A0AEC0";
  }
}

interface FlowNode extends Node {
  style?: React.CSSProperties;
}

const getLayoutedElements = (nodes: FlowNode[], edges: Edge[], direction: 'LR' | 'TB' = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction, nodesep: 100, ranksep: 120 });

  nodes.forEach((node) => {
    const height = node.data?.estimatedHeight || node.style?.minHeight || 80;
    dagreGraph.setNode(node.id, { label: node.data.label, width: node.style?.width ?? 180, height: height });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.position = {
      x: nodeWithPosition.x - (node.style?.width as number ?? 180) / 2,
      y: nodeWithPosition.y - (node.style?.minHeight as number ?? 80) / 2,
    };
    
    node.sourcePosition = Position.Right;
    node.targetPosition = Position.Left;
  });

  return { nodes, edges };
};

// アニメーション枠線用のCSSとスタイル
const animatedBorderStyle = {
  animation: 'border-dance 1s linear infinite',
  border: '6px solid #facc15', // より太い黄色枠線
  background: '#fef9c3', // 薄い黄色
  zIndex: 10,
};

const styleSheet = `
@keyframes border-dance {
  0% { box-shadow: 0 0 0 0 #facc15; }
  50% { box-shadow: 0 0 0 12px #fde047; }
  100% { box-shadow: 0 0 0 0 #facc15; }
}
`;

export default function DashboardPage() {
  const state = useAgentState();
  const [layoutedNodes, setLayoutedNodes] = React.useState<FlowNode[]>([]);
  const [layoutedEdges, setLayoutedEdges] = React.useState<Edge[]>([]);

  useEffect(() => {
    if (!state || !Array.isArray(state.history)) {
       setLayoutedNodes([]);
       setLayoutedEdges([]);
       return;
    }

    try {
      const nodes: FlowNode[] = [];
      const edges: Edge[] = [];
      const nodeMap = new Map<string, { node: FlowNode }>();
      const baseNodeStyle = { 
        borderRadius: 6, 
        padding: '10px 15px', 
        fontSize: '20px', 
        whiteSpace: 'pre-wrap',
        fontWeight: 500,
        width: 180,
      };
      
      // アクション履歴の前後関係を記録するための変数
      let lastActionNodeId: string | null = null;
      
      // 仮説ごとの最後のアクションノードを記録
      const lastActionNodeByHypothesis: { [hypId: string]: string } = {};
      // 仮説ごとの最初のアクションノードを記録
      const firstActionNodeByHypothesis: { [hypId: string]: string } = {};
      // generate_hypothesis ノードと生成された仮説の対応
      const generateHypNodes: { nodeId: string, mode: string, parentId: string | null }[] = [];
      
      // 主要なアクションとその表示名
      const actionDisplayNames: {[key: string]: string} = {
        'eda': 'EDA',
        'generate_hypothesis': 'Generate hyp',
        'evaluate_hypothesis': 'Evaluate',
        'query_data_agent': 'query_data_agent',
        'decision_maker': 'Decision'
      };
      
      // 表示対象のアクションタイプを定義（decisionを除外）
      const displayedActionTypes = ['eda', 'generate_hypothesis', 'evaluate_hypothesis', 'query_data_agent', 'refine_hypothesis'];
      
      // 最後のアクションノードIDを特定
      let latestActionNodeId: string | null = null;
      for (let i = state.history.length - 1; i >= 0; i--) {
        const entry = state.history[i];
        if (entry?.type === 'node' && displayedActionTypes.includes(entry.content?.name)) {
          latestActionNodeId = `action-${entry.content.name}-${i}`;
          break;
        }
      }
      
      // メインアクション履歴を表示（アクションノード生成）
      state.history.forEach((entry: any, index: number) => {
        if (!entry || !entry.type) return;
        
        // 必要なアクションのみ表示
        let actionName = "";
        let nodeId = `hist-${index}`;
        let nodeLabel = "";
        let style: React.CSSProperties = { ...baseNodeStyle };
        
        if (entry.type === 'node' && entry.content?.name) {
          actionName = entry.content.name;
          
          // 表示対象アクションの場合のみノードを生成（decisionは除外）
          if (displayedActionTypes.includes(actionName)) {
            let evaluationReason = '';
            nodeLabel = actionDisplayNames[actionName] || actionName;
            nodeId = `action-${actionName}-${index}`;
            
            let nodeSpecificData = {}; 
            let estimatedNodeHeight = Number(style.minHeight) || 80; // デフォルトの高さ

            if (nodeId === latestActionNodeId) {
              style = {
                ...style,
                ...animatedBorderStyle,
              };
            }

            if (actionName === 'query_data_agent') {
              style.background = '#6b21a8'; 
              const linkedAgents = entry.content?.process;
              if (linkedAgents && Array.isArray(linkedAgents)) {
                nodeSpecificData = { ...nodeSpecificData, linkedAgents: linkedAgents };
                let linkedAgentsHeight = 20; // 「連携エージェント:」ラベル分の高さ
                if (linkedAgents.length > 0) {
                  linkedAgentsHeight += linkedAgents.length * 45; // 1エージェントあたりの高さを35から45に増やす
                }
                estimatedNodeHeight = (Number(style.minHeight) || 80) + linkedAgentsHeight; 
              }
            } else if (actionName === 'evaluate_hypothesis'){
              // evaluate_hypothesisの色設定などはここで行う (現状はelseで青)
              // 必要であれば、estimatedNodeHeightも内容に応じて調整
              style.background = '#2563eb'; 
            } else {
              style.background = '#2563eb'; 
            }
            style.color = 'white';
            style.width = 180;
            style.minHeight = 80;
            
            // evaluate_hypothesis の場合、結果を表示する
            if (actionName === 'evaluate_hypothesis') {
              const focusHypId = entry.currently_investigating_hypothesis_id;
              if (focusHypId) {
                // 対応する observation を探す
                const observationEntry = state.history.find((obsEntry: any, obsIndex: number) => 
                  obsIndex > index && 
                  obsEntry.type === 'observation' && 
                  obsEntry.content?.hypothesis_id === focusHypId
                );
                
                if (observationEntry && observationEntry.content) {
                  const status = observationEntry.content.status || 'N/A';
                  const reason = observationEntry.content.evaluation_reason || '';
                  const shortReason = reason.substring(0, 50) + (reason.length > 50 ? '...' : '');
                  evaluationReason = reason; // ここで理由を格納
                  // ラベルに結果を追加
                  nodeLabel += `\n(${status})`;
                  // 理由も表示する場合は追加
                  // nodeLabel += `\n${shortReason}`;
                  // ステータスに応じて背景色を変更
                  if (status === 'supported') {
                    style.background = '#10B981'; // 緑
                  } else if (status === 'rejected') {
                    style.background = '#718096'; // 赤
                  } else if (status === 'inconclusive') {
                     style.background = '#718096'; // 灰色
                  } else if (status === 'needs_revision') {
                     style.background = '#f59e0b'; // 黄色
                  }
                }
              }
            }
            
            // generate_hypothesis ノードの生成モードを記録
            if (actionName === 'generate_hypothesis' && entry.content) {
              const mode = entry.content.mode || 'initial';
              const parentId = entry.content.parent_id || null;
              
              // ノードラベルにモード情報を追加
              nodeLabel = `Generate hyp\n(${mode})`;
              
              // 生成情報を記録
              generateHypNodes.push({
                nodeId: nodeId,
                mode: mode,
                parentId: parentId
              });
            }
            // generate_hypothesis ノードの生成モードを記録
            if (actionName === 'refine_hypothesis' && entry.content) {
              const parentId = entry.content.parent_id || null;
              
              // ノードラベルにモード情報を追加
              nodeLabel = `Refine hyp`;
              
              // 生成情報を記録
              generateHypNodes.push({
                nodeId: nodeId,
                mode: 'refine',
                parentId: parentId
              });
            }
            
            // 現在のフォーカス仮説ID
            const focusHypId = entry.currently_investigating_hypothesis_id || null;
            
            const newNode: FlowNode = {
              id: nodeId,
              data: { 
                label: nodeLabel,
                focusHypId: focusHypId, 
                evaluationReason: evaluationReason,
                estimatedHeight: estimatedNodeHeight, // 推定高さをdataに含める
                ...nodeSpecificData 
              },
              position: { x: 0, y: 0 }, 
              style: actionName === 'query_data_agent' ? {} : style, // queryDataAgentNodeの場合は空のスタイルを渡し、他は既存のstyleを維持
              type: actionName === 'query_data_agent' ? 'queryDataAgentNode' : 'action', 
              sourcePosition: Position.Right, 
              targetPosition: Position.Left,  
            };
            
            nodes.push(newNode);
            nodeMap.set(nodeId, { node: newNode });
            
            // フォーカス仮説に基づく接続ロジック
            if (focusHypId) {
              // この仮説の最初のアクションノードを記録（まだない場合）
              if (!firstActionNodeByHypothesis[focusHypId]) {
                firstActionNodeByHypothesis[focusHypId] = nodeId;
              }
              
              // フォーカス仮説がある場合、同じ仮説の前のノードと接続
              if (lastActionNodeByHypothesis[focusHypId]) {
                edges.push({
                  id: `e-flow-hyp-${focusHypId}-${lastActionNodeByHypothesis[focusHypId]}-${nodeId}`,
                  source: lastActionNodeByHypothesis[focusHypId],
                  target: nodeId,
                  type: 'straight',
                  style: { stroke: '#888', strokeWidth: 2 }
                });
              }
              // この仮説の最後のノードを更新
              lastActionNodeByHypothesis[focusHypId] = nodeId;
            } else {
              // フォーカス仮説がない場合、仮説なしの前のノードと接続
              if (lastActionNodeId) {
                edges.push({
                  id: `e-flow-general-${lastActionNodeId}-${nodeId}`,
                  source: lastActionNodeId,
                  target: nodeId,
                  type: 'straight',
                  style: { stroke: '#888', strokeWidth: 2 }
                });
              }
              // 仮説なしの最後のノードを更新
              lastActionNodeId = nodeId;
            }
          }
        }
      });
      
      // 仮説ノードを生成
      if (state.current_hypotheses && Array.isArray(state.current_hypotheses)) {
        state.current_hypotheses.forEach((hyp: any) => {
          if (!hyp || !hyp.id) return;
          
          const hypNodeId = `hyp-${hyp.id}`;
          // IF-THEN形式の表示
          const shortText = hyp.text ? hyp.text.substring(0, 100) + (hyp.text.length > 100 ? '...' : '') : '';
          let hypLabel = `仮説：${hyp.id}\npriority：${hyp.priority}`;
          hypLabel += `\n${shortText}`;
          
          // 仮説の色分け（supported: 緑, rejected: 灰色, デフォルト: 薄い青）
          let bgColor = '#93c5fd'; // デフォルト薄い青
          if (hyp.status === 'supported') {
            bgColor = '#38A169'; // 緑：サポート済み
          } else if (hyp.status === 'rejected') {
            bgColor = '#718096'; // 灰色：棄却済み
          } else if (hyp.status === 'needs_revision') {
            bgColor = '#718096'; // 灰色：棄却済みと同様の色
          } else if (hyp.id.startsWith('sup_')) {
            bgColor = '#718096'; // 灰色：補助仮説
          }
          
          // 高さを固定 200px に設定（推定ロジックは使用しない）
          const estimatedHeight = 80; // 変更後の高さ (60px -> 80px)

          const hypStyle: React.CSSProperties = {
             ...baseNodeStyle,
             background: bgColor,
             color: 'white',
             width: 260,
             minHeight: estimatedHeight, // 変更後の高さを適用
             border: hyp.id.startsWith('sup_') ? '2px solid yellow' : 'none',
             borderRadius: '40px', // 高さと幅に応じて調整 (高さ80pxなら40pxでカプセル型に近い)
          };

          const newNode: FlowNode = {
            id: hypNodeId,
            data: { label: `${hyp.id}`, id: hyp.id, estimatedHeight: estimatedHeight }, // estimatedHeight を更新
            position: { x: 0, y: 0 }, // 後でdagreで配置
            style: hypStyle,
            type: 'hypothesis',
            sourcePosition: Position.Right, // 右から出る
            targetPosition: Position.Left,  // 左から入る
          };
          
          nodes.push(newNode);
          nodeMap.set(hypNodeId, { node: newNode });
        });
      }
      
      // 各仮説の最初のアクションノードに、仮説ノードを接続
      Object.entries(firstActionNodeByHypothesis).forEach(([hypId, firstActionNodeId]) => {
        const hypNodeId = `hyp-${hypId}`;
        if (nodeMap.has(hypNodeId) && nodeMap.has(firstActionNodeId)) {
          edges.push({
            id: `e-hyp-first-${hypId}-${firstActionNodeId}`,
            source: hypNodeId,
            target: firstActionNodeId,
            type: 'straight',
            style: { stroke: '#888', strokeWidth: 2 }
          });
        }
      });
      
      // Generate Hypothesisノードと生成された仮説ノードを接続
      if (state.current_hypotheses && Array.isArray(state.current_hypotheses)) {
        generateHypNodes.forEach(genNode => {
          state.current_hypotheses.forEach((hyp: any) => {
            if (!hyp || !hyp.id) return;
            
            // 生成モードと親IDに基づいて、このGenerateノードで生成された仮説かどうかを判定
            let isGenerated = false;
            if (genNode.mode === 'initial' && !hyp.parent_hypothesis_id && !hyp.id.startsWith('sup_')) {
              // 初期仮説の場合（親なし、非サブ仮説）
              isGenerated = true;
            } else if (genNode.mode === 'supporting' && hyp.parent_hypothesis_id === genNode.parentId && hyp.id.startsWith('sub_')) {
              // サポート仮説の場合（指定された親ID、かつsub_プレフィックス）
              isGenerated = true;
            } else if (genNode.mode === 'refine' && hyp.parent_hypothesis_id === genNode.parentId && !hyp.id.startsWith('sup_')) {
              // refine_hypothesisノードで生成された仮説（親IDが一致、sup_でない）
              isGenerated = true;
            }
            
            if (isGenerated) {
              const hypNodeId = `hyp-${hyp.id}`;
              if (nodeMap.has(hypNodeId)) {
                // Generate/Refine Hypothesisノードから仮説ノードへのエッジを追加
                edges.push({
                  id: `e-gen-${genNode.nodeId}-${hypNodeId}`,
                  source: genNode.nodeId,
                  target: hypNodeId,
                  type: 'straight',
                  style: { stroke: '#888', strokeWidth: 2 }
                });
              }
            }
          });
        });
      }

      console.log("[Layout] Calculating layout with dagre for", nodes.length, "nodes and", edges.length, "edges.");
      // 横方向（左→右）レイアウトを明示し、エッジタイプを滑らかなベジエ曲線（default）にする
      edges.forEach(e => { e.type = 'default'; });
      let { nodes: dagreLayoutedNodes, edges: dagreLayoutedEdges } = getLayoutedElements(nodes, edges, 'LR');
      
      // --- ここからY軸中心を揃える処理を追加 ---
      const nodesById = new Map(dagreLayoutedNodes.map(n => [n.id, n]));

      // 仮説IDごとにノードをグループ化 (focusHypIdを持つノード)
      const nodesByHypothesis: { [hypId: string]: FlowNode[] } = {};
      dagreLayoutedNodes.forEach(node => {
        const focusHypId = node.data?.focusHypId;
        if (focusHypId && typeof focusHypId === 'string') {
          if (!nodesByHypothesis[focusHypId]) {
            nodesByHypothesis[focusHypId] = [];
          }
          nodesByHypothesis[focusHypId].push(node);
        }
      });

      // 各仮説グループごとにY軸中心を揃える
      Object.values(nodesByHypothesis).forEach(group => {
        if (group.length > 0) {
          // グループ内のノードのY座標の中心の平均値を計算
          let sumOfCenterY = 0;
          group.forEach(node => {
            const nodeHeight = node.data?.estimatedHeight || node.style?.minHeight || 80;
            sumOfCenterY += (node.position.y + nodeHeight / 2); // 各ノードの中心Yを合計
          });
          const averageCenterY = sumOfCenterY / group.length;

          // 各ノードのY位置を調整
          group.forEach(node => {
            const nodeHeight = node.data?.estimatedHeight || node.style?.minHeight || 80;
            node.position.y = averageCenterY - (nodeHeight / 2);
          });
        }
      });
      
      // 仮説IDを持たないノード群 (メインフローなど) についても同様に処理 (今回は簡略化のため未実装)
      // もしメインフローも同様に揃えたい場合は、別途グルーピングと処理が必要

      setLayoutedNodes(dagreLayoutedNodes);
      setLayoutedEdges(dagreLayoutedEdges);

    } catch (error) {
      console.error("[Flow Generation/Layout] Error:", error);
      setLayoutedNodes([]);
      setLayoutedEdges([]);
    }

  }, [state]);

  if (!state) {
    console.log("[DashboardPage] State is null/undefined, rendering spinner.");
    return (
      <Center h="100vh">
        <Spinner />
      </Center>
    );
  }

  const agentsToShow = state?.available_data_agents_and_skills ?? [];

  return (
    <>
      <Box as="style">{styleSheet}</Box>
      <Grid templateColumns="3fr 1fr" gap={4} p={4} h="100vh">
        <GridItem border="1px solid #E2E8F0" borderRadius="md" p={2} overflow="hidden" minHeight="200px">
          <Heading size="sm" mb={2}>
            Execution Flow & Hypotheses
          </Heading>
          <ExecutionGraph nodes={layoutedNodes} edges={layoutedEdges} />
        </GridItem>
        <GridItem border="1px solid #E2E8F0" borderRadius="md" p={2} overflowY="auto" minHeight="200px" display="flex" flexDirection="column">
          <Heading size="sm" mb={2}>
            Connected Agents
          </Heading>
          <AgentTable agents={agentsToShow} />
          <Box mt={8} flexGrow={1} display="flex" flexDirection="column" justifyContent="flex-start">
            <Divider my={4} />
            <Box>
              <Heading size="xs" mb={2}>仮説一覧</Heading>
              <Table size="xs" variant="simple">
                <Thead>
                  <Tr>
                    <Th minWidth="60px">仮説ID</Th>
                    <Th minWidth="80px">status</Th>
                    <Th>内容</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {state.current_hypotheses && state.current_hypotheses.map((hyp: any) => (
                    <Tr key={hyp.id}>
                      <Td minWidth="60px" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{hyp.id}</Td>
                      <Td minWidth="80px" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{hyp.status}</Td>
                      <Td style={{whiteSpace: 'pre-wrap'}}>{hyp.text}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </Box>
          </Box>
        </GridItem>
      </Grid>
    </>
  );
} 