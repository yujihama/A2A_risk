import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Text, Badge, VStack } from '@chakra-ui/react';

interface QueryDataAgentNodeData {
  label: string;
  linkedAgents?: string[];
  evaluationReason?: string; // 既存のActionNode.tsxのdata構造を参考に含める
  focusHypId?: string;     // 同上
}

// NodePropsからstyleを除外し、代わりにidやselectedなどのプロパティは利用可能
const QueryDataAgentNode: React.FC<NodeProps<QueryDataAgentNodeData>> = ({ data, id, selected, dragging }) => {
  const nodeStyle = {
    background: '#6b21a8', // 紫色
    color: 'white',
    padding: '10px 15px',
    borderRadius: '6px',
    width: 'auto', // 内容に合わせて幅を自動調整
    minWidth: '180px', // 最小幅
    // border: selected ? '2px solid #82c9ff' : '1px solid #4a0072', // 一時的にコメントアウト
    // boxShadow: dragging ? '0 4px 8px rgba(0,0,0,0.3)' : '0 2px 4px rgba(0,0,0,0.2)', // コメントアウト継続
  };

  return (
    <Box style={nodeStyle}> 
      <Handle type="target" position={Position.Left} style={{ background: '#555' }} />
      <VStack spacing={2} align="stretch">
        <Text fontWeight="bold" textAlign="center">{data.label}</Text>
        {data.linkedAgents && data.linkedAgents.length > 0 && (
          <Box 
            mt={2} 
            p={2} 
            bg="rgba(255,255,255,0.1)" 
            borderRadius="md" 
            width="100%" // 親要素(VStack)の幅に合わせる
            maxWidth="160px" // 最大幅を設定 (180px - padding*2 程度)
          >
            <Text fontSize="sm" mb={1} fontWeight="semibold">連携エージェント:</Text>
            <VStack spacing={1} align="stretch">
              {data.linkedAgents.map((agentName, index) => (
                <Badge 
                  key={index} 
                  colorScheme="teal" 
                  variant="solid"
                  px={2}
                  py={0.5}
                  borderRadius="sm"
                  textAlign="left" // 折り返しを考慮して左揃えに
                  fontSize="xs"
                  whiteSpace="normal" // テキストの折り返しを許可
                  wordBreak="break-word" // 単語の途中でも折り返し
                  display="block" // Badgeをブロック要素にして幅と折り返しを有効に
                  w="100%" // 親の幅いっぱいに広がるように
                >
                  {agentName}
                </Badge>
              ))}
            </VStack>
          </Box>
        )}
      </VStack>
      <Handle type="source" position={Position.Right} style={{ background: '#555' }} />
    </Box>
  );
};

export default QueryDataAgentNode; 