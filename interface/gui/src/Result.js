import React, { useState, useEffect} from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import {github} from 'react-syntax-highlighter/dist/esm/styles/hljs';
import styled from "styled-components";

function Result(props){
    return(
        <div className="pr-5 pb-2">
            <hr/>       
            <br/>
            <ResultCode {...props}/>
            
            <div className="pb-0" style={{fontSize:'1em'}}>
                <span>{props.identifier.split(".")[0]}</span>
                .
                <span style={{fontWeight:'bold'}}>{props.identifier.split(".")[1]}</span>
            </div>
            <div className="pb-2" style={{
                fontSize:'.8em'
            }}>
                {props.nwo}
            </div>
            {props.docstring && <ResultDescription {...props}/>}
             {!props.docstring && <div style={{fontStyle:"italic", opacity:"70%"}}>No summary</div>}
        </div>
    )
}


function ResultCode(props){
    const style1={maxHeight:'150px', overflow:"hidden", width:"100%"}
    const style2={maxHeight:'500px', overflow:"scroll", width:"100%"}
    const [codeOpen, setCodeOpen] = useState(false);
    
    useEffect(() => {
      setCodeOpen(false)
    }, [props.identifier]);
    
    return(
        <div style={{position:"relative", width:"100%", boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"}}>
            <SyntaxHighlighter 
                language="java" 
                style={github} 
                showLineNumbers
                showInlineLineNumbers
                customStyle={codeOpen?style2:style1}
            >
                {props.function+"\n"}
            </SyntaxHighlighter>
            <div 
                className="py-1 px-2"
                onClick={()=>setCodeOpen(!codeOpen)}
                style={{position:"absolute", bottom:0, textAlign:"right", width:"100%", fontSize:".9em", color:"#0000ee", cursor:"pointer", background:(codeOpen?"":"linear-gradient(0deg, rgba(255,255,255,1) 0%, rgba(255,255,255,0.7469362745098039) 65%, rgba(255,255,255,0) 100%)")}}
            >

                    Show {codeOpen?"Less":"More"}
            </div>
            
        </div>
    )
}


function ResultDescription(props){
    const [textOpen, setTextOpen] = useState(false);
    
    useEffect(() => {
      setTextOpen(props.docstring_summary===props.docstring)
    }, [props.docstring_summary, props.docstring]);
    
    return(
        <div 
            className="pt-2"
            style={{fontSize:"1.1em", width:"100%"}}
        >
            <Description className={textOpen?"":"clamped"} style={{overflowWrap: "break-word", whiteSpace: "pre-wrap"}}>
                {props.docstring}
            </Description>
            
            {(props.docstring_summary!==props.docstring) &&
                <div 
                    onClick={()=>setTextOpen(!textOpen)}
                    style={{fontSize:".9em", color:"#0000ee", cursor:"pointer"}}
                 >
                 Read {textOpen?"Less":"More"}
                </div>
            }
        </div>
    )
}


const Description = styled.div`
  display: -webkit-box;
  -webkit-box-orient: vertical;  
`


export default Result