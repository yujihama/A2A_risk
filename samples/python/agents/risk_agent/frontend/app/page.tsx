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
  dagreGraph.setGraph({ rankdir: direction, nodesep: 60, ranksep: 45 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { label: node.data.label, width: node.style?.width ?? 180, height: node.style?.minHeight ?? 80 });
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
        fontSize: '16px', 
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
      const displayedActionTypes = ['eda', 'generate_hypothesis', 'evaluate_hypothesis', 'query_data_agent'];
      
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
            
            // アクションタイプごとに色分け
            if (actionName === 'query_data_agent') {
              style.background = '#6b21a8';
              // どのデータエージェントを呼び出したかをラベルに追加
              const agentId = entry.content?.parameters?.agent_skill_id ?? entry.content?.agent_id ?? '';
              if (agentId) {
                nodeLabel += `\n(${agentId})`;
              }
            } else {
              style.background = '#2563eb'; // その他のアクションは青色
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
            
            // 現在のフォーカス仮説ID
            const focusHypId = entry.currently_investigating_hypothesis_id || null;
            
            const newNode: FlowNode = {
              id: nodeId,
              data: { 
                label: nodeLabel,
                focusHypId: focusHypId, // フォーカス仮説ID情報を保持
                evaluationReason: evaluationReason // 理由を追加
              },
              position: { x: 0, y: 0 }, // 後でdagreで配置
              style: style,
              type: 'action',
              sourcePosition: Position.Right, // 右から出る
              targetPosition: Position.Left,  // 左から入る
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
          } else if (hyp.id.startsWith('sup_')) {
            bgColor = '#718096'; // 灰色：補助仮説
          }
          
          // 高さを固定 200px に設定（推定ロジックは使用しない）
          const estimatedHeight = 200;

          const hypStyle: React.CSSProperties = {
             ...baseNodeStyle,
             background: bgColor,
             color: 'white',
             width: 260,
             minHeight: estimatedHeight,
             border: hyp.id.startsWith('sup_') ? '2px solid yellow' : 'none',
          };

          const newNode: FlowNode = {
            id: hypNodeId,
            data: { label: hypLabel },
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
            }
            
            if (isGenerated) {
              const hypNodeId = `hyp-${hyp.id}`;
              if (nodeMap.has(hypNodeId)) {
                // Generate Hypothesisノードから仮説ノードへのエッジを追加
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
      const { nodes: layoutedNodesResult, edges: layoutedEdgesResult } = getLayoutedElements(nodes, edges, 'LR');
      
      // --- ここから仮説ごとにノードの中心を揃える処理 ---
      // focusHypId（仮説ID）ごとにノードをグループ化
      const nodesByHypothesis: { [hypId: string]: FlowNode[] } = {};
      layoutedNodesResult.forEach(node => {
        const hypId = node.data?.focusHypId || node.data?.parent_hypothesis_id || null;
        if (hypId) {
          if (!nodesByHypothesis[hypId]) nodesByHypothesis[hypId] = [];
          nodesByHypothesis[hypId].push(node);
        }
      });

      // ↓ 中心で揃えるロジック + 仮説ノード中心に合わせてシフト
      Object.entries(nodesByHypothesis).forEach(([hypId, group]) => {
        if (group.length > 0) {
          // 対応する仮説ノードの中心Yを取得
          const hypNodeId = `hyp-${hypId}`;
          const hypNodeEntry = nodeMap.get(hypNodeId);
          if (hypNodeEntry) {
            const hypNode = hypNodeEntry.node;
            const hypHeight = Number(hypNode.style?.minHeight ?? 80);
            const hypCenterY = hypNode.position.y + hypHeight / 2;

            // グループの各ノードの中心Yを仮説中心に合わせる
            group.forEach(node => {
              const nodeHeight = Number(node.style?.minHeight ?? 80);
              node.position.y = hypCenterY - nodeHeight / 2;
            });
          }
        }
      });
      // --- ここまで ---

      setLayoutedNodes(layoutedNodesResult);
      setLayoutedEdges(layoutedEdgesResult);

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
            <Table size="sm" variant="simple">
              <Thead>
                <Tr>
                  <Th>仮説ID</Th>
                  <Th>priority</Th>
                  <Th>status</Th>
                  <Th>内容</Th>
                </Tr>
              </Thead>
              <Tbody>
                {state.current_hypotheses && state.current_hypotheses.map((hyp: any) => (
                  <Tr key={hyp.id}>
                    <Td>{hyp.id}</Td>
                    <Td>{hyp.priority}</Td>
                    <Td>{hyp.status}</Td>
                    <Td style={{whiteSpace: 'pre-wrap', maxWidth: 400}}>{hyp.text}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </Box>
        </Box>
      </GridItem>
    </Grid>
  );
} 