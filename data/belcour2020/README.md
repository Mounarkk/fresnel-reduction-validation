This repository contains a mitsuba implementation for "Bringing an Accurate
Fresnel to Real-Time Rendering: a Preintegrable Decomposition" by
Laurent Belcour, Megane Bati, and Pascal Barla.

Sources to the mitsuba plugin are located in the `plugins` directory.
A dummy scene in the `scene` directory is provided. You can use it using
the following call to mitsuba:

 $ mitsuba -Dfresnel=FRESNEL_OURS -Dmaterial=Au sphere.xml
 
 You can replace FRESNEL_OURS by (FRESNEL_REFERENCE or FRESNEL_SCHLICK) to
 compare the new model to mitsuba's implementation and Schlick's approximation.
 Different materials can be selected by replacing Au with other mitsuba
 IOR list.
