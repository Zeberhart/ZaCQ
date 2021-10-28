import React from 'react';


function PrioritizeMessage(props){
    
    function CQSentence(setValues){
        if("verb" in setValues){
            return  "Prioritizing functions that "+[setValues.verb, setValues.direct_object_modifiers, setValues.direct_object, setValues.preposition, setValues.preposition_object_modifiers, setValues.preposition_object].join(" ").trim()+". "
        }else if("object" in setValues){
            return  "Prioritizing functions related to "+[setValues.object_modifiers, setValues.object].join(" ").trim()+". "
        }
    }
    
    function KWSentence(intents){
        if(intents){
            return "Prioritizing functions related to "+ intents.join(", ") +". "
        }
    }
    
    const emptySentence = <span style={{color:"#aa2222"}}>Couldn't find any functions matching those criteria. Try going back or entering a new query.</span>
    
    let sentence = "";
    if(!props.numCandidates){
         sentence = emptySentence
    }else if(Object.keys(props.setValues).length>0){
         sentence = CQSentence(props.setValues);
    }else if(props.intents.length>0){
         sentence = KWSentence(props.intents);
    }
    
    if(props.active){
        return(
            <div style={{fontSize:".9em", opacity:.8}}>
                <span 
                    className='mx-2'
                    style={{cursor:"pointer", color:"#0000ee"}}
                    onClick={props.undoRefinement}
                >
                    Undo  
                </span>
                | 
                <span 
                    className='mx-2'
                    style={{cursor:"pointer", color:"#0000ee"}}
                    onClick={props.resetRefinement}
                >
                    Reset  
                </span>  
                {sentence && <span>|</span>}
                <span 
                    className='mx-2'
                    style={{fontStyle:"italic"}}
                >
                   {sentence} 
                </span>
                  
           </div>
        )
    }else{
        return(
            <br/>
        )
    }
}

export default PrioritizeMessage