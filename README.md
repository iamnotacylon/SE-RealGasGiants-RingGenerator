# SE-RealGasGiants-RingGenerator
 This is a program, set of instructions, and mod skeleton to create ring textures usable with the Real Gas Giants mod for Space Engineers (https://steamcommunity.com/workshop/filedetails/?id=3232085677).

# Create the ring textures
 To use this generator, make sure you have the relevant python3 packages installed, and run ring_gen.py with python. There are many settings tunable at the top of this file, and you can obviously adjust it how you want for your own purposes, but we think the default ones are usually pretty good.
 
 Running the program will generate 5 png files of ring textures for Real Gas Giants to UV map. If you don't like all or some of them, they can be re-rolled by running the program again. Any that you mostly like and think you can hand-tune the rest of the way can be edited in your favorite graphics editor.

 When you're ready to finalize your images into a mod usable in space engineers with RealGasGiants, simply open the images you like with Paint.NET (https://www.getpaint.net/) and "Save As" a .dds file; make sure to select "BC7 (sRGB, DX 11+)", "Cube Map from crossed image", "Generate Mip Maps", "Bicubic (Smooth)", and "Use gamma correction" selected. 
 
 While other, but not all, dds generation settings will work, we have found this to work as well as the others, and it's easier to simply have one set of instructions. We recommend paint.net over the image converter distributed with space engineers simply because there's nothing to learn about using it first, and you don't need to use the windows command line, which many dislike.

# Create the mod
 As explained by the author of Real Gas Giants (https://steamcommunity.com/workshop/filedetails/discussion/3232085677/4337607287061689717/), the next step is setting your textures up with a publishable mod so that it is actually usable. 

 To do so, begin by copying the provided mod skeleton folder to your AppData/Roaming/SpaceEngineers/Mods folder. This will turn it into a local mod. You should also adjust the toplevel name here.

 Next, move your dds ring texture files into the Textures folder. 

 From there, edit the provided TransparentMaterials.sbc and GasGiantDefault.sbc according to the instructions in the Ring Skin Steps: at https://steamcommunity.com/workshop/filedetails/discussion/3232085677/4337607287061689717/
