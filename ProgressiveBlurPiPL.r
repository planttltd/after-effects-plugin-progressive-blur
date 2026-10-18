#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "AE_General.r"

resource 'PiPL' (16000) {
	{
		Kind { AEEffect },
		Name { "Progressive Blur" },
		Category { "Blur & Sharpen" },
		CodeMacARM64 { "EffectMain" },
		AE_PiPL_Version { 2, 0 },
		AE_Effect_Spec_Version { PF_PLUG_IN_VERSION, PF_PLUG_IN_SUBVERS },
		AE_Effect_Version { 524289 },	/* PF_VERSION(1,0,0,DEVELOP,1) — match my_version */
		AE_Effect_Info_Flags { 0 },
		AE_Effect_Global_OutFlags { 0x02000000 },	/* DEEP_COLOR_AWARE */
		AE_Effect_Global_OutFlags_2 { 0x0A001400 },	/* THREADED | GPU_F32 | FLOAT_COLOR | SMART_RENDER */
		AE_Effect_Match_Name { "ROYP Progressive Blur" },
		AE_Reserved_Info { 0 },
		AE_Effect_Support_URL { "https://royplant.com" }
	}
};
